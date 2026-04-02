import asyncio
import re
import sys
import os
import json
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator import Generator
from components.orchestration.nodes import extract_filters_node
from components.utils import _acall_hf_endpoint, getconfig
from components.retriever.filters import FILTER_VALUES

from test_questions import standalone_questions, history_blocks, safeguard_questions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Parse config (same logic as main.py)
_config = getconfig("params.cfg")
MAX_TURNS = _config.getint("conversation_history", "MAX_TURNS", fallback=3)
_filterable_fields_raw = _config.get("metadata_filters", "filterable_fields", fallback="")
FILTERABLE_FIELDS: Dict[str, str] = {}
for _item in _filterable_fields_raw.split(","):
    _item = _item.strip()
    if ":" in _item:
        _name, _ftype = _item.split(":", 1)
        FILTERABLE_FIELDS[_name.strip()] = _ftype.strip()
    elif _item:
        FILTERABLE_FIELDS[_item] = "str"


@dataclass
class EvalCase:
    question: str
    subset: str                              # "standalone" | "history" | "safeguard"
    user_messages_history: Optional[str]     # pre-built history string, None if not applicable


def build_eval_suite() -> List[EvalCase]:
    """Flatten the three test_questions subsets into a unified list of EvalCase objects.

    History blocks: only the last turn is evaluated; prior turns (sliced to MAX_TURNS)
    are passed as user_messages_history — mirroring what ui_adapters does in production.
    """
    cases = []

    for q in standalone_questions:
        cases.append(EvalCase(question=q, subset="standalone", user_messages_history=None))

    for block in history_blocks:
        if len(block) < 2:
            continue
        current_query = block[-1]
        prior_turns = block[:-1][-MAX_TURNS:]
        history_str = "\n".join(f"USER: {t}" for t in prior_turns)
        cases.append(EvalCase(question=current_query, subset="history", user_messages_history=history_str))

    for q in safeguard_questions:
        cases.append(EvalCase(question=q, subset="safeguard", user_messages_history=None))

    return cases


def _result_path(base: str, filtered: bool) -> str:
    suffix = "_filtered" if filtered else ""
    return os.path.join(RESULTS_DIR, f"{base}{suffix}.json")


async def evaluate_questions(
    cases: List[EvalCase],
    retriever,
    generator=None,
    filterable_fields: Dict[str, str] = None,
    filter_values: Dict[str, list] = None,
) -> List[Dict]:
    """Run retrieval for each case. If generator is provided, extract filters first."""
    eval_data = []

    for case in cases:
        print(f"🧐 [{case.subset}] Processing: {case.question[:50]}...")

        # --- Filter extraction (filters mode only) ---
        filters = None
        if generator is not None:
            state = {"query": case.question, "user_messages_history": case.user_messages_history}
            result_state = await extract_filters_node(
                state,
                generator=generator,
                filterable_fields=filterable_fields,
                filter_values=filter_values,
            )
            filters = result_state.get("metadata_filters")
            print(f"   Filters extracted: {filters}")

        # --- Embed ---
        embed_res = await _acall_hf_endpoint(
            retriever.embedding_endpoint_url,
            retriever.hf_token,
            {"inputs": case.question},
        )
        query_vector = embed_res[0]

        # --- Stage 1: vector search ---
        raw_candidates = await retriever._asearch_qdrant(query_vector, filters=filters)

        # --- Stage 2: full pipeline (rerank) ---
        retriever_kwargs = {"filters": filters} if filters else {}
        final_docs = await retriever.ainvoke(case.question, **retriever_kwargs)

        entry = {
            "question": case.question,
            "subset": case.subset,
            "stage1_raw_vector_results": [
                {"content": c.get("answer"), "score": c.get("score"), "metadata": c.get("answer_metadata")}
                for c in raw_candidates
            ],
            "stage2_reranked_results": [
                {"content": d.page_content, "rerank_score": d.metadata.get("rerank_score"), "metadata": d.metadata}
                for d in final_docs
            ],
        }
        if generator is not None:
            entry["filters_applied"] = filters

        eval_data.append(entry)

    return eval_data


async def run_retrieval_only(filters_enabled: bool):
    all_questions = standalone_questions + [b[-1] for b in history_blocks if len(b) >= 2] + safeguard_questions
    if not all_questions or all(len(q.split()) <= 1 for q in all_questions):
        print("💥 test_questions.py still has placeholder content.")
        print("   Edit tests/eval/test_questions.py and add real questions before running eval.")
        sys.exit(1)

    print("🚀 Initializing Retriever...")
    try:
        retriever = create_retriever_from_config("params.cfg")
    except Exception as e:
        print(f"💥 Failed to load config/retriever: {e}")
        sys.exit(1)

    generator = None
    if filters_enabled:
        if not FILTERABLE_FIELDS:
            print("💥 --filters passed but filterable_fields is empty in params.cfg. Aborting.")
            sys.exit(1)
        print("🔍 Filter extraction enabled. Initializing Generator for filter extraction...")
        generator = Generator()

    cases = build_eval_suite()
    results = await evaluate_questions(
        cases,
        retriever,
        generator=generator,
        filterable_fields=FILTERABLE_FIELDS if filters_enabled else None,
        filter_values=FILTER_VALUES if filters_enabled else None,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = _result_path("retrieval_eval_results", filters_enabled)

    print("💾 Exporting results...")
    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", indent=4, force_ascii=False)

    print(f"✅ Retrieval complete! Results saved to {output_path}")
    sys.exit(0)


async def get_judge_verdict(generator: Generator, query: str, content: str, metadata: dict):
    from langchain_core.documents import Document
    context_as_docs = [Document(page_content=content, metadata=metadata)]

    judge_query = (
        f"[SYSTEM EVALUATION]\n"
        f"Task: Judge if the provided Context & Source answer the Question.\n"
        f"Question: \"{query}\"\n\n"
        f"Response Format:\nREASON: [Short reasoning in English]\nVERDICT: YES or VERDICT: NO (exactly as written, no other characters)"
    )

    response = await generator.generate(
        query=judge_query,
        context=context_as_docs,
        chatui_format=False
    )
    return response


def _verdict_is_yes(judge_output: str) -> bool:
    """Extract YES/NO verdict robustly, stripping markdown formatting and collapsing whitespace."""
    clean = re.sub(r"[*_`#]", "", judge_output.upper())
    clean = re.sub(r"\s+", " ", clean)
    match = re.search(r"VERDICT\s*:\s*(YES|NO)", clean)
    return bool(match and match.group(1) == "YES")


async def run_evaluation_batch(filters_enabled: bool, input_file=None):
    if input_file is None:
        input_file = _result_path("retrieval_eval_results", filters_enabled)

    generator_instance = Generator()
    output_filename = _result_path("judged_eval_report", filters_enabled)

    with open(input_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    final_report = []
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            final_report = json.load(f)
        print(f"🔄 Resuming from checkpoint. {len(final_report)} questions already judged.")

    processed_questions = {entry['question'] for entry in final_report}

    for entry in source_data:
        user_q = entry['question']
        if user_q in processed_questions:
            continue

        print(f"⚖️ Judging results for: {user_q[:40]}...")

        for doc in entry['stage2_reranked_results']:
            judge_output = await get_judge_verdict(
                generator_instance,
                user_q,
                doc['content'],
                doc['metadata']
            )
            doc['judge_raw_output'] = judge_output
            doc['is_relevant'] = _verdict_is_yes(judge_output)

        final_report.append(entry)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)

        print(f"💾 Checkpoint saved. Progress: {len(final_report)}/{len(source_data)}")

    print(f"✅ Success! Full report finalized at {output_filename}")
    sys.exit(0)


async def run_sample_eval(filters_enabled: bool, input_file=None):
    if input_file is None:
        input_file = _result_path("retrieval_eval_results", filters_enabled)

    generator_instance = Generator()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_data = data[:2]
    final_report = []

    print(f"🚀 Starting Sample Run: Evaluating {len(sample_data)} queries...")

    for entry in sample_data:
        user_q = entry['question']
        print(f"\n🔍 Query: {user_q}")

        for i, doc in enumerate(entry['stage2_reranked_results']):
            judge_output = await get_judge_verdict(
                generator_instance,
                user_q,
                doc['content'],
                doc['metadata']
            )

            is_relevant = _verdict_is_yes(judge_output)
            doc['judge_raw_output'] = judge_output
            doc['is_relevant'] = is_relevant

            status = "✅ RELEVANT" if is_relevant else "❌ IRRELEVANT"
            print(f"  - Doc {i+1}: {status}")
            print(f"    Reasoning: {judge_output.split('REASON:')[1].split('VERDICT:')[0].strip() if 'REASON:' in judge_output else 'N/A'}")

        final_report.append(entry)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_filename = _result_path("sample_judged_report", filters_enabled)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print(f"\n✨ Sample complete. Check '{output_filename}' for details.")
    sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChaBo RAG Evaluation")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "batch", "sample"],
        default="retrieval",
        help="retrieval: run Stage 1 and save results | batch: judge with LLM (resumes from checkpoint) | sample: judge first 2 questions only"
    )
    parser.add_argument(
        "--filters",
        action="store_true",
        default=False,
        help="Enable metadata filter extraction before retrieval. Results saved with '_filtered' suffix for comparison."
    )
    args = parser.parse_args()

    modes = {
        "retrieval": lambda: run_retrieval_only(args.filters),
        "batch": lambda: run_evaluation_batch(args.filters),
        "sample": lambda: run_sample_eval(args.filters),
    }
    asyncio.run(modes[args.mode]())
