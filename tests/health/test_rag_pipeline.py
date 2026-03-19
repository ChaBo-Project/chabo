import asyncio
import logging
import datetime
import os
import sys

# Add src/ to path so imports match how main.py resolves them
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from test_components import test_retriever_unit, run_full_pipeline_test


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"rag_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


logger = logging.getLogger("RAG_Evaluator")


async def run_single_test(case_name, query, retriever, generator):
    """Executes one test case and logs the results."""
    logger.info(f"▶️ STARTING TEST: {case_name}")
    logger.info(f"Query: {query}")

    result = await run_full_pipeline_test(query, retriever, generator)

    if not result["success"]:
        logger.error(f"❌ TEST FAILED: {case_name} | Error: {result.get('error')}")
        return False

    # Per-doc score analysis
    docs = result["docs"]
    logger.info(f"✅ Retrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        score = doc.metadata.get('rerank_score', 'N/A')
        logger.info(f"  [{i}] Score: {score} | Snippet: {doc.page_content[:60]}...")

    logger.info(f"📝 Final Response (first 100 chars): {result['answer'][:100]}...")

    # Hallucination risk check — only meaningful when docs were returned
    if docs:
        avg_score = sum(d.metadata.get('rerank_score', 0) for d in docs) / len(docs)
        if avg_score < 0.2 and len(result['answer']) > 300:
            logger.warning("⚠️ High Risk: LLM gave a long answer despite very low retrieval scores.")

    return True


async def main():
    setup_logging()

    from components.retriever.retriever_orchestrator import create_retriever_from_config
    from components.generator.generator_orchestrator import Generator

    retriever = create_retriever_from_config("params.cfg")
    generator = Generator()

    # Add your test queries here
    test_cases = [
        ("Sample query", "What is the purpose of this system?"),
    ]

    results = []
    for case_name, query in test_cases:
        success = await run_single_test(case_name, query, retriever, generator)
        results.append((case_name, success))

    logger.info("--- TEST SUMMARY ---")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")


if __name__ == "__main__":
    asyncio.run(main())
