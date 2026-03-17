import logging
import asyncio

logger = logging.getLogger("RAG_Test_Suite")

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

            elif event_type == "sources":
                captured_sources = data.get("sources", [])
                logger.info(f"📍 Sources Event: Received {len(captured_sources)} citations.")

        # 3. Validation & Reporting
        logger.info("✅ Full Pipeline Test Complete!")
        logger.info(f"📊 Summary: {len(accumulated_response)} chars generated | {len(captured_sources)} sources.")

        return {
            "answer": accumulated_response,
            "sources": captured_sources,
            "success": True
        }

    except Exception as e:
        logger.error(f"❌ Full Pipeline Test Failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
