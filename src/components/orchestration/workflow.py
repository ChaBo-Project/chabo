from langgraph.graph import StateGraph, START, END
from .nodes import retrieve_node, generate_node_streaming
from functools import partial
from .state import GraphState

def simple_graph(retriever_svc, generator_svc):
    """Builds the compiled LangGraph."""
    workflow = StateGraph(GraphState) # Or your GraphState class

    # Injecting services into nodes
    r_node = partial(retrieve_node, retriever=retriever_svc)
    g_node = partial(generate_node_streaming, generator=generator_svc)

    workflow.add_node("retrieve", r_node)
    workflow.add_node("generate", g_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()