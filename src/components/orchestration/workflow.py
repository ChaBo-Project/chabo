"""
LangGraph Workflow Setup for ChaBo RAG Orchestrator
"""
import logging
from functools import partial
from typing import Dict
from langgraph.graph import StateGraph, START, END

from .state import GraphState
from .nodes import retrieve_node, generate_node_streaming, ingest_node, extract_filters_node

logger = logging.getLogger(__name__)


def build_workflow(retriever_instance, generator_instance, filterable_fields: Dict[str, str] = None, filter_values: Dict[str, list] = None):
    """
    Build and compile the LangGraph workflow for RAG orchestration.

    Args:
        retriever_instance: Initialised ChaBoHFEndpointRetriever
        generator_instance: Initialised Generator
        filterable_fields: Dict of {field_name: type} for LLM-based metadata filter extraction.
                           Pass {} or None to disable (extract_filters node becomes a pass-through).
        filter_values: Dict of {field_name: [valid_values]} for constrained LLM extraction.
                       Every field in filterable_fields must have an entry here.
    """
    if filterable_fields is None:
        filterable_fields = {}
    if filter_values is None:
        filter_values = {}

    workflow = StateGraph(GraphState)

    # Inject services into nodes
    r_node = partial(retrieve_node, retriever=retriever_instance)
    g_node = partial(generate_node_streaming, generator=generator_instance)
    f_node = partial(extract_filters_node, generator=generator_instance, filterable_fields=filterable_fields, filter_values=filter_values)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract_filters", f_node)
    workflow.add_node("retrieve", r_node)
    workflow.add_node("generate", g_node)

    # Define edges: ingest -> extract_filters -> retrieve -> generate
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "extract_filters")
    workflow.add_edge("extract_filters", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    compiled_graph = workflow.compile()

    logger.info(f"LangGraph workflow compiled (filterable_fields={list(filterable_fields.keys())}, filter_values_fields={list(filter_values.keys())})")
    return compiled_graph