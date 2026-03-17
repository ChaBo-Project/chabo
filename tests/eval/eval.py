import asyncio
import sys
import os
import json
import pandas as pd
from typing import List, Dict
from langchain_core.documents import Document

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator import Generator
from components.utils import _acall_hf_endpoint

from test_questions import questions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


async def evaluate_questions(questions: List[str], retriever) -> List[Dict]:
    eval_data = []
    for q in questions:
        print(f"🧐 Processing: {q[:50]}...")

        # 1. Get Embedding Vector
        embed_payload = {"inputs": q}
        embed_res = await _acall_hf_endpoint(retriever.embedding_endpoint_url, retriever.hf_token, embed_payload)
        query_vector = embed_res[0]

        # 2. Stage 1: Vector Search (Candidate Retrieval)
        raw_candidates = await retriever._asearch_qdrant(query_vector)

        # 3. Stage 2: Full Pipeline (Reranked)
        final_docs = await retriever.ainvoke(q)

        eval_data.append({
            "question": q,
            "stage1_raw_vector_results": [
                {"content": c.get("answer"), "score": c.get("score"), "metadata": c.get("answer_metadata")}
                for c in raw_candidates
            ],
            "stage2_reranked_results": [
                {"content": d.page_content, "rerank_score": d.metadata.get("rerank_score"), "metadata": d.metadata}
                for d in final_docs
            ]
        })
    return eval_data


async def run_retrieval_only():
    print("🚀 Initializing Retriever...")
    try:
        retriever = create_retriever_from_config("params.cfg")
    except Exception as e:
        print(f"💥 Failed to load config/retriever: {e}")
        sys.exit(1)

    results = await evaluate_questions(questions, retriever)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "retrieval_eval_results.json")

    print("💾 Exporting results...")
    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", indent=4, force_ascii=False)

    print(f"✅ Retrieval complete! Results saved to {output_path}")
    sys.exit(0)


async def get_judge_verdict(generator: Generator, query: str, content: str, metadata: dict):
    context_as_docs = [Document(page_content=content, metadata=metadata)]

    judge_query = (
        f"[SYSTEM EVALUATION]\n"
        f"Task: Judge if the provided Context & Source answer the Question.\n"
        f"Question: \"{query}\"\n\n"
        f"Response Format:\nREASON: [Short reasoning in English]\nVERDICT: [YES or NO]"
    )

    response = await generator.generate(
        query=judge_query,
        context=context_as_docs,
        chatui_format=False
    )
    return response


async def run_evaluation_batch(input_file=None):
    if input_file is None:
        input_file = os.path.join(RESULTS_DIR, "retrieval_eval_results.json")

    generator_instance = Generator()
    output_filename = os.path.join(RESULTS_DIR, "judged_eval_report.json")

    # 1. Health Check
    if not await generator_instance.validate_health():
        print("❌ Model health check failed. Batch aborted.")
        return

    # 2. Load the source retrieval data
    with open(input_file, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    # 3. Load existing progress if it exists (checkpoint)
    final_report = []
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            final_report = json.load(f)
        print(f"🔄 Resuming from checkpoint. {len(final_report)} questions already judged.")

    processed_questions = {entry['question'] for entry in final_report}

    # 4. Main Judging Loop
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
            doc['is_relevant'] = "VERDICT: YES" in judge_output.upper()

        final_report.append(entry)

        # 5. Save checkpoint after every question
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=4)

        print(f"💾 Checkpoint saved. Progress: {len(final_report)}/{len(source_data)}")

    print(f"✅ Success! Full report finalized at {output_filename}")
    sys.exit(0)


async def run_sample_eval(input_file=None):
    if input_file is None:
        input_file = os.path.join(RESULTS_DIR, "retrieval_eval_results.json")

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

            is_relevant = "VERDICT: YES" in judge_output.upper()
            doc['judge_raw_output'] = judge_output
            doc['is_relevant'] = is_relevant

            status = "✅ RELEVANT" if is_relevant else "❌ IRRELEVANT"
            print(f"  - Doc {i+1}: {status}")
            print(f"    Reasoning: {judge_output.split('REASON:')[1].split('VERDICT:')[0].strip() if 'REASON:' in judge_output else 'N/A'}")

        final_report.append(entry)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_filename = os.path.join(RESULTS_DIR, "sample_judged_report.json")
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
    args = parser.parse_args()

    modes = {
        "retrieval": run_retrieval_only,
        "batch": run_evaluation_batch,
        "sample": run_sample_eval,
    }
    asyncio.run(modes[args.mode]())
