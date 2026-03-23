"""
ChaBo RAG Orchestrator - Production Entry Point
"""
import os
import logging
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
import uvicorn
from functools import partial

from components.retriever.retriever_orchestrator import create_retriever_from_config
from components.generator.generator_orchestrator import Generator
from components.orchestration.workflow import build_workflow
from components.orchestration.ui_adapters import chatui_adapter, chatui_file_adapter
from components.orchestration.state import ChatUIInput, ChatUIFileInput
from components.utils import getconfig
from components.retriever.filters import FILTER_VALUES
from typing import Dict

config = getconfig("params.cfg")
MAX_TURNS = config.getint("conversation_history", "MAX_TURNS", fallback=3)
MAX_CHARS = config.getint("conversation_history", "MAX_CHARS", fallback=8000)

# Parse filterable_fields: "field:type,field:type" → {"field": "type"}
_filterable_fields_raw = config.get("metadata_filters", "filterable_fields", fallback="")
FILTERABLE_FIELDS: Dict[str, str] = {}
for _item in _filterable_fields_raw.split(","):
    _item = _item.strip()
    if ":" in _item:
        _name, _ftype = _item.split(":", 1)
        FILTERABLE_FIELDS[_name.strip()] = _ftype.strip()
    elif _item:
        FILTERABLE_FIELDS[_item] = "str"  # default to str if no type declared

# Validate: every field declared in params.cfg must have valid values in filters.py
if FILTERABLE_FIELDS:
    _missing = [f for f in FILTERABLE_FIELDS if f not in FILTER_VALUES]
    if _missing:
        raise ValueError(
            f"Fields declared in params.cfg [metadata_filters] are missing from filters.py: {_missing}. "
            "Add valid values for these fields in src/components/retriever/filters.py "
            "or remove them from filterable_fields in params.cfg."
        )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize services
logger.info("Initializing ChaBoHFEndpointRetriever and Generator...")
retriever_instance = create_retriever_from_config(config_file="params.cfg")
generator_instance = Generator()

# Build the LangGraph workflow
compiled_graph = build_workflow(retriever_instance, generator_instance, filterable_fields=FILTERABLE_FIELDS, filter_values=FILTER_VALUES)


#----------------------------------------
# FASTAPI SETUP
#----------------------------------------

app = FastAPI(title="ChaBo RAG Orchestrator", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "ChaBo RAG Orchestrator API",
        "endpoints": {
            "health": "/health",
            "chatfed-ui-stream": "/chatfed-ui-stream (LangServe)",
            "chatfed-with-file-stream": "/chatfed-with-file-stream (LangServe)",
        }
    }


#----------------------------------------
# LANGSERVE ROUTES
#----------------------------------------

# Inject compiled_graph and config into adapters
text_adapter = partial(chatui_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS)
file_adapter = partial(chatui_file_adapter, compiled_graph=compiled_graph, max_turns=MAX_TURNS, max_chars=MAX_CHARS)

# Text-only endpoint
add_routes(
    app,
    RunnableLambda(text_adapter),
    path="/chatfed-ui-stream",
    input_type=ChatUIInput,
    output_type=str,
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

# File upload endpoint
add_routes(
    app,
    RunnableLambda(file_adapter),
    path="/chatfed-with-file-stream",
    input_type=ChatUIFileInput,
    output_type=str,
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

logger.debug(f"ChatUIInput schema: {ChatUIInput.model_json_schema()}")
logger.debug(f"ChatUIFileInput schema: {ChatUIFileInput.model_json_schema()}")

#----------------------------------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))

    logger.info(f"Starting ChaBo RAG Orchestrator on {host}:{port}")
    logger.info(f"API Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)