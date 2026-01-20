"""
LangGraph orchestration nodes for retrieval and generation

NEEDS TO BE UPDATED
""" 
import logging
logger = logging.getLogger(__name__)
from datetime import datetime
import json
from typing import TYPE_CHECKING, Dict, Any, List
from langchain_core.documents import Document
from .telemetry import extract_retriever_telemetry

# Assuming these Type definitions are available from state.py and retriever_orchestrator.py
if TYPE_CHECKING:
    from components.retriever.retriever_orchestrator import ChaBoHFEndpointRetriever
    from components.generator.generator_orchestrator import Generator
    from components.orchestration.state import GraphState



async def retrieve_node(
    state: 'GraphState',
    retriever: 'ChaBoHFEndpointRetriever' # Injected LangChain BaseRetriever instance
    ) -> 'GraphState':
    """
    Node to retrieve relevant context using the ChaBoHFEndpointRetriever.
    The retriever performs Embed -> Search -> Rerank in one async call.
    """

    start_time = datetime.now()

    # 1. Extract Query and Filters
    filters = state.get("metadata_filters")
    metadata = state.get("metadata", {})
    logger.info(f"Retrieval: {state['query'][:50]}...")

    raw_documents: list[Document] = []

    try:
        retriever_kwargs = {}
        if filters:
            retriever_kwargs['filters'] = filters

        raw_documents = await retriever.ainvoke(
            input=state['query'],
            **retriever_kwargs
        )

        duration = (datetime.now() - start_time).total_seconds()
        retriever_config = {
            "initial_k": retriever.initial_k,
            "final_k": retriever.final_k,
            "qdrant_mode": retriever.qdrant_mode,
        }

        retriever_telemetry = extract_retriever_telemetry(raw_documents, retriever_config)

        metadata.update({
            "retrieval_duration": duration,
            "filters_applied": json.dumps(filters) if filters else "None",
            "retriever_config": retriever_telemetry,
            "retrieval_success": True
        })
        return {
            "raw_documents": raw_documents,
            "metadata": metadata
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)

        metadata.update({
            "retrieval_duration": duration,
            "retrieval_success": False,
            "retrieval_error": str(e)
        })

        return {"raw_documents": [], "metadata": metadata}
    

async def generate_node_streaming(state: "GraphState", generator: "Generator", *, writer):
    """
    Node to generate the final response with StreamWriter for LangGraph custom streaming.
    Uses StreamWriter to emit events that LangGraph can capture with stream_mode="custom".
    """
    start_time = datetime.now()

    query = state.get("query")
    raw_docs = state.get("raw_documents", [])
    metadata = state.get("metadata", {})

    accumulated_text = ""
    logger.info(f"Generation: {query[:50]}... ({len(raw_docs)} docs)")
    conversation_context = state.get("conversation_context")

    try:
        async for event in generator.generate_streaming(
            query=query,
            context=raw_docs,
            chatui_format=True,
            conversation_context=conversation_context
        ):
            # Track content to calculate metadata (length) at the end
            if event.get("event") == "data":
                accumulated_text += event.get("data", "")

            # Use StreamWriter to emit custom events
            writer(event)

        # Final Telemetry Update
        duration = (datetime.now() - start_time).total_seconds()
        metadata.update({
            "generation_duration": duration,
            "generation_success": True,
            "response_length": len(accumulated_text)
        })

        logger.info(f"Streaming complete in {duration:.2f}s. Length: {len(accumulated_text)}")
        logger.debug(f"Final answer: {accumulated_text}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Generation node failed: {e}", exc_info=True)
        metadata.update({
            "generation_duration": duration,
            "generation_success": False,
            "generation_error": str(e)
        })
        writer({"event": "error", "data": {"error": str(e)}})



# from .state import GraphState


# if TYPE_CHECKING:
#     from components.retriever.retriever_orchestrator import RetrieverOrchestrator
#     from components.orchestration.state import GraphState

# async def retrieve_node(
#     state: GraphState, 
#     retriever: 'RetrieverOrchestrator' # Injected service instance
#     ) -> GraphState:
#     """Retrieve relevant context using adapter"""
    
#     start_time = datetime.now()
#     logger.info(f"Retrieval: {state['query'][:50]}...")
#     context = ""

#     try:
#         # Get filters from state (provided by ChatUI or LLM agent)
#         filters = state.get("metadata_filters")
        
#         # --- FILLED CODE START ---
        
#         # Call the async method on the injected service instance
#         # The retriever orchestrator handles the remote API call to the Reranker/Embedder service
        
#         context_docs, retriever_meta = await retriever.aretrieve(
#             query=latest_message,
#             filters=filters
#         )
        
#         # Format the retrieved documents into a single context string 
#         # (This is commonly done here or inside the orchestrator)
#         context = "\n---\n".join([doc.page_content for doc in context_docs])
        
#         # --- FILLED CODE END ---
        
#         duration = (datetime.now() - start_time).total_seconds()
#         metadata = state.get("metadata", {})
        
#         # Update metadata and append retriever-specific metadata
#         metadata.update({
#             "retrieval_duration": duration,
#             "context_length": len(context) if context else 0,
#             "retrieval_success": True,
#             "filters_applied": filters,
#             "retriever_config": retriever_meta, # Add metadata from retriever call
#         })
        
#         # Return the updated state
#         return {"context": context, "metadata": metadata}
    
#     except Exception as e:
#         # ... (Error handling logic is good, no change needed) ...
#         duration = (datetime.now() - start_time).total_seconds()
#         logger.error(f"Retrieval failed: {str(e)}")
        
#         metadata = state.get("metadata", {})
#         metadata.update({
#             "retrieval_duration": duration,
#             "retrieval_success": False,
#             "retrieval_error": str(e)
#         })
#         # Note: We return context as an empty string on failure to avoid cascading errors
#         return {"context": "", "metadata": metadata}


# async def retrieve_node(state: GraphState) -> GraphState:
#     """Retrieve relevant context using adapter"""
#     start_time = datetime.now()
#     logger.info(f"Retrieval: {state['query'][:50]}...")
    
#     try:
#         # Get filters from state (provided by ChatUI or LLM agent)
#         filters = state.get("metadata_filters")
        
#         # instantiate the retirever instance
#         # get context using aysnc call
        
        
#         duration = (datetime.now() - start_time).total_seconds()
#         metadata = state.get("metadata", {})
#         metadata.update({
#             "retrieval_duration": duration,
#             "context_length": len(context) if context else 0,
#             "retrieval_success": True,
#             "filters_applied": filters,
#             "retriever_config": # get metadata from retirever
#         })
        
#         return {"context": context, "metadata": metadata}
    
#     except Exception as e:
#         duration = (datetime.now() - start_time).total_seconds()
#         logger.error(f"Retrieval failed: {str(e)}")
        
#         metadata = state.get("metadata", {})
#         metadata.update({
#             "retrieval_duration": duration,
#             "retrieval_success": False,
#             "retrieval_error": str(e)
#         })
#         return {"context": "", "metadata": metadata}