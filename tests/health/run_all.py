"""
run_all.py — ChaBo Health Check Suite

Runs connectivity checks for each service in order, then executes the
retriever unit test and full RAG pipeline test.

Usage (from repo root):
    source chabo_env/bin/activate
    python tests/health/run_all.py
"""

import asyncio
import logging
import datetime
import os
import sys

# Ensure repo root is on the path so src/ imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from test_components import test_retriever_unit, run_full_pipeline_test


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"health_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


logger = logging.getLogger("HealthCheck")

SAMPLE_QUERY = "What is the purpose of this system?"


async def check_qdrant():
    """Verify Qdrant is reachable and the configured collection exists."""
    try:
        from src.components.utils import getconfig
        from qdrant_client import QdrantClient

        url = getconfig("qdrant", "url")
        port = getconfig("qdrant", "port")
        collection = getconfig("qdrant", "collection")
        api_key = os.getenv("QDRANT_API_KEY")

        client = QdrantClient(url=f"{url}:{port}", api_key=api_key)
        collections = [c.name for c in client.get_collections().collections]

        if collection not in collections:
            logger.error(f"❌ Qdrant: collection '{collection}' not found. Available: {collections}")
            return False

        logger.info(f"✅ Qdrant: reachable, collection '{collection}' exists.")
        return True

    except Exception as e:
        logger.error(f"❌ Qdrant check failed: {str(e)}", exc_info=True)
        return False


async def check_embedding():
    """Verify the embedding endpoint returns a valid vector."""
    try:
        import httpx
        from src.components.utils import getconfig

        url = getconfig("hf_endpoints", "embedding_endpoint_url")
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": "health check"}

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        logger.info(f"✅ Embedding endpoint: reachable, status {response.status_code}.")
        return True

    except Exception as e:
        logger.error(f"❌ Embedding endpoint check failed: {str(e)}", exc_info=True)
        return False


async def check_reranker():
    """Verify the reranker endpoint returns scores."""
    try:
        import httpx
        from src.components.utils import getconfig

        url = getconfig("hf_endpoints", "reranker_endpoint_url")
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"query": "health check", "texts": ["sample document"]}

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        logger.info(f"✅ Reranker endpoint: reachable, status {response.status_code}.")
        return True

    except Exception as e:
        logger.error(f"❌ Reranker endpoint check failed: {str(e)}", exc_info=True)
        return False


async def main():
    setup_logging()
    logger.info("=" * 50)
    logger.info("ChaBo Health Check Suite")
    logger.info("=" * 50)

    results = {}

    # --- Connectivity checks ---
    logger.info("\n--- Step 1: Connectivity Checks ---")
    results["Qdrant"] = await check_qdrant()
    results["Embedding Endpoint"] = await check_embedding()
    results["Reranker Endpoint"] = await check_reranker()

    if not all(results.values()):
        logger.error("❌ One or more connectivity checks failed. Fix the above errors before proceeding.")
    else:
        # --- Component checks (only if connectivity passed) ---
        logger.info("\n--- Step 2: Component Checks ---")
        from src.components.retriever.retriever_orchestrator import Retriever
        from src.components.generator.generator_orchestrator import Generator

        retriever = Retriever()
        generator = Generator()

        retriever_result = await test_retriever_unit(SAMPLE_QUERY, retriever)
        results["Retriever Unit"] = retriever_result["success"]

        pipeline_result = await run_full_pipeline_test(SAMPLE_QUERY, retriever, generator)
        results["RAG Pipeline"] = pipeline_result["success"]

    # --- Summary ---
    logger.info("\n--- HEALTH CHECK SUMMARY ---")
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✅ All checks passed. ChaBo is ready.")
    else:
        logger.error("\n❌ Some checks failed. Review logs above.")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
