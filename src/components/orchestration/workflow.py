"""
LangGraph Workflow Setup for ChaBo RAG Orchestrator
"""
import logging
from functools import partial
from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import retrieve_node, generate_node_streaming, ingest_node

logger = logging.getLogger(__name__)


def build_workflow(retriever_instance, generator_instance):
    """
    Build and compile the LangGraph workflow for RAG orchestration
    """
    workflow = StateGraph(GraphState)

    # Inject services into nodes (ingest_node doesn't need dependency injection)
    r_node = partial(retrieve_node, retriever=retriever_instance)
    g_node = partial(generate_node_streaming, generator=generator_instance)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("retrieve", r_node)
    workflow.add_node("generate", g_node)

    # Define edges: ingest -> retrieve -> generate
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    compiled_graph = workflow.compile()

    logger.info("LangGraph workflow compiled successfully with ingest node")
    return compiled_graph