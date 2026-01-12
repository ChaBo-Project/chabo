import logging
import sys
import uvicorn
import asyncio
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import json
# 1. Logging Setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator  import Generator
from components.orchestration.telemetry import extract_retriever_telemetry
from components.orchestration.state import ChatUIInput

try:
    logger.info("Initializing ChaBoHFEndpointRetriever and Generator...")
    ## NOTE: Ensure your params.cfg file exists and contains the necessary details
    retriever_instance = create_retriever_from_config(config_file="params.cfg")
    generator_instance = Generator()
except Exception as e:
    logger.error(f"Failed to initialize Service: {e}")


# --- 2. FastAPI App Setup ---
app = FastAPI(title="ChaBo RAG Orchestrator API")

@app.get("/health")
async def health_check():
    """Verify the service is alive."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "Orchestrator is running",
        "docs": "/docs",
        "health": "/health",
        "endpoint": "/chatfed-ui-stream/"
    }

@app.on_event("startup")
async def startup_event():
    """
    Runs automatically when the Hugging Face Space starts.
    """
    logger.info("🚀 Space started....Rnnning Test.... ")
    
    # Run the test in the background so it doesn't block the server startup
    asyncio.create_task(test_retriever())

async def test_retriever():
    try:
        query = "What is EUDR and deforestation projects?"
        logger.info(f"⏳ Calling Retirever ainvoke...{query}")
        docs = await retriever_instance.ainvoke(query)
    
        if not docs:
            # Check logs to see if it was a search fail (ERROR) or just no matches (INFO)
            logger.warning("⚠️ No documents were returned (either service error or no semantic matches).")
        else:
            logger.info(f"✅ Found {len(docs)} docs.")
        # Proceed to generation...

    except Exception as e:
        # This only triggers if something completely unhandled happens (like a syntax error)
        logger.error(f"❌ Unhandled Pipeline Crash: {str(e)}", exc_info=True)


# --- 3. Execution ---
if __name__ == "__main__":
    # Run using uvicorn
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info", access_log=True)