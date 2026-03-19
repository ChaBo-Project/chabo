import logging
import asyncio
import os

logger = logging.getLogger("RAG_Test_Suite")


async def check_embedding(retriever):
    """Verify the embedding endpoint returns a valid vector using the retriever's own config."""
    try:
        from components.utils import _acall_hf_endpoint
        payload = {"inputs": "health check"}
        result = await _acall_hf_endpoint(retriever.embedding_endpoint_url, retriever.hf_token, payload)
        if not isinstance(result, list) or not result:
            raise ValueError("Unexpected response shape from embedding endpoint")
        logger.info(f"✅ Embedding endpoint: reachable, vector dim={len(result[0])}.")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding endpoint check failed: {str(e)}", exc_info=True)
        return False


async def check_reranker(retriever):
    """Verify the reranker endpoint returns scores using the retriever's own config."""
    try:
        from components.utils import _acall_hf_endpoint
        payload = {"query": "health check", "texts": ["sample document"]}
        result = await _acall_hf_endpoint(retriever.reranker_endpoint_url, retriever.hf_token, payload)
        if not isinstance(result, list) or 'score' not in result[0]:
            raise ValueError("Unexpected response shape from reranker endpoint")
        logger.info(f"✅ Reranker endpoint: reachable, returned {len(result)} score(s).")
        return True
    except Exception as e:
        logger.error(f"❌ Reranker endpoint check failed: {str(e)}", exc_info=True)
        return False


async def check_qdrant(retriever):
    """Verify Qdrant is reachable and the configured collection exists using the retriever's own client."""
    try:
        if retriever.qdrant_mode.lower() == 'native':
            client = await retriever._aget_qdrant_client()
            collections = [c.name for c in (await client.get_collections()).collections]
            if retriever.qdrant_collection not in collections:
                logger.error(f"❌ Qdrant: collection '{retriever.qdrant_collection}' not found. Available: {collections}")
                return False
            logger.info(f"✅ Qdrant: reachable, collection '{retriever.qdrant_collection}' exists.")
        elif retriever.qdrant_mode.lower() == 'gradio':
            # Triggering _aget_qdrant_client forces GradioClient init — success means space is reachable
            await retriever._aget_qdrant_client()
            logger.info(f"✅ Qdrant (Gradio): space reachable at {retriever.qdrant_url}.")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant check failed: {str(e)}", exc_info=True)
        return False


async def test_retriever_unit(query, retriever_instance):
    """
    Unit test for the Retriever only.
    Verifies connectivity, scoring, and document structure.
    Returns a dict with success, docs, and optional error.
    """
    try:
        logger.info(f"⏳ [Retriever Unit] Calling ainvoke: '{query}'")
        docs = await retriever_instance.ainvoke(query)

        if not docs:
            logger.warning("⚠️ No documents returned. Check endpoint or query relevance.")
            return {"success": False, "docs": [], "error": "No documents returned"}

        logger.info(f"✅ Found {len(docs)} docs.")

        top_score = docs[0].metadata.get('rerank_score')
        if top_score is not None:
            logger.info(f"🔝 Top Rerank Score: {top_score} (reranker active)")
        else:
            logger.warning("⚠️ No rerank_score in metadata — reranker may not have run")

        return {"success": True, "docs": docs}

    except Exception as e:
        logger.error(f"❌ Retriever Unit Crash: {str(e)}", exc_info=True)
        return {"success": False, "docs": [], "error": str(e)}


async def run_full_pipeline_test(query, retriever_instance, generator_instance):
    """
    Integration test for the full RAG pipeline.
    Steps: Retrieval -> Streaming Generation -> Source Attribution.
    """
    try:
        # 1. Step: Retrieval
        logger.info(f"🚀 [Pipeline Test] Step 1: Retrieving context for '{query}'...")
        docs = await retriever_instance.ainvoke(query)

        if not docs:
            logger.warning("⚠️ No context found. Generator will run in 'no-context' mode.")

        # 2. Step: Manual Streaming Generation
        logger.info("🚀 [Pipeline Test] Step 2: Starting streaming generation...")
        accumulated_response = ""
        captured_sources = []

        async for event in generator_instance.generate_streaming(
            query=query,
            context=docs,
            chatui_format=True
        ):
            event_type = event.get("event")
            data = event.get("data", "")

            if event_type == "data":
                accumulated_response += data
                if len(accumulated_response) % 100 < len(data):
                    logger.info(f"  ...Streaming: {len(accumulated_response)} chars")

            elif event_type == "final_answer":
                captured_sources = data.get("webSources", [])
                logger.info(f"📍 Sources received: {len(captured_sources)} citations.")

        # 3. Validation & Reporting
        logger.info("✅ Full Pipeline Test Complete!")
        logger.info(f"📊 Summary: {len(accumulated_response)} chars generated | {len(captured_sources)} sources.")

        return {
            "answer": accumulated_response,
            "sources": captured_sources,
            "docs": docs,
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Full Pipeline Test Failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
