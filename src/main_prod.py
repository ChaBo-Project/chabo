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

# Step 2 Run manual pipeline with instances directly
async def run_manual_pipeline(data):
    if isinstance(data, dict):
        text_value = data.get('text', '')
        messages_value = data.get('messages', None)
        preprompt_value = data.get('preprompt', None)
    else:
        text_value = getattr(data, 'text', '')
        messages_value = getattr(data, 'messages', None)
        preprompt_value = getattr(data, 'preprompt', None)
        
    # Extract query
        
    # Convert dict messages to objects if needed
    messages = []
    for msg in messages_value:
        if isinstance(msg, dict):
            messages.append(type('Message', (), {
                'role': msg.get('role', 'unknown'),
                'content': msg.get('content', '')
            })())
        else:
            messages.append(msg)

    logger.info(f"Context: {messages}")
    # Extract latest user query
    user_messages = [msg for msg in messages if msg.role == 'user']
    query = user_messages[-1].content if user_messages else text_value
#  Step 2.1: Manual Retrieval
    logger.info(f"Step 2.1: Retrieving context for '{query}'...")
    start_time = datetime.now()
    filters = None
    try: 
        docs = await retriever_instance.ainvoke(query)
        logger.info(f"✅ Found {len(docs)} docs.")

        duration = (datetime.now() - start_time).total_seconds()
        retriever_config = {
            "initial_k": retriever_instance.initial_k,
            "final_k": retriever_instance.final_k,
            "qdrant_mode": retriever_instance.qdrant_mode,
        }
        
        retriever_telemetry = extract_retriever_telemetry(docs, retriever_config)
        metadata = {
            "retrieval_duration": duration,
            "filters_applied": json.dumps(filters) if filters else "None", 
            "retriever_config": retriever_telemetry,     
            "retrieval_success": True
        }
        logger.debug(f"Retrieval info: {metadata}")
    except Exception as e:
        logger.error("Retrieval failed")
        logger.error(f"Error: {e}")
        docs = []

# Step 2.2: Manual Streaming Generation
    logger.info(f"Step 2.2 Generation: {query}")
    accumulated_text = ""
    captured_sources = None
    try:
        async for event in generator_instance.generate_streaming(
            query=query,
            context=docs,
            chatui_format=True
        ):

            if event.get("event") == "data":
                chunk = event.get("data")
                accumulated_text += chunk
                yield chunk
                await asyncio.sleep(0.01)
            elif event.get("event") == "sources":
                captured_sources = event.get("data", {}).get("sources", [])
                # payload = {"sources": captured_sources}
                logging.info(f"Captured Sources: {captured_sources}")
                # yield f"event: sources\ndata: {json.dumps(payload)}\n\n"
            elif event.get("event") == "end":
                if captured_sources:
                    sources_text = "\n\n**Sources:**\n"
                    for i, source in enumerate(captured_sources, 1):
                        if isinstance(source, dict):
                            title = source.get('title', 'Unknown')
                            link = source.get('link', '#')
                            if link in ['','#']:
                                link = f"doc://{title}"
                            sources_text += f"{i}. [{title}]({link})\n"
                            # sources_text += f"{i}. [{source.get('title', 'Unknown')}]({source.get('link', '#')})\n"
                        else:
                            sources_text += f"{i}. {source}\n"
                    logging.info(f"Soruces text:{sources_text}")
                        
                    yield sources_text
            # await asyncio.sleep(0.01)

            # Final Telemetry Update
        duration = (datetime.now() - start_time).total_seconds()
        metadata.update({
                "generation_duration": duration,
                "generation_success": True,
                "response_length": len(accumulated_text)
            })
            
        logger.info(f"Streaming complete in {duration:.2f}s. Length: {len(accumulated_text)}")
        logger.debug(f"Generator _ Retrieval info: {metadata}")
        logger.debug(f"Final answer: {accumulated_text}")
    except Exception as e:
        logger.error(f"Generation node failed: {e}")
        yield f"Error: {str(e)}"

# --- 3. FastAPI App Setup ---
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

add_routes(
    app,
    RunnableLambda(run_manual_pipeline),
    path="/chatfed-ui-stream",
    input_type=ChatUIInput,
    output_type=str, # Tells LangServe to expect string yields
)

# --- 3. Execution ---
if __name__ == "__main__":
    # Run using uvicorn
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info", access_log=True)