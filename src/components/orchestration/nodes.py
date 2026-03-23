"""
LangGraph orchestration nodes for retrieval and generation

NEEDS TO BE UPDATED
"""
import logging
logger = logging.getLogger(__name__)
from datetime import datetime
import json
from typing import TYPE_CHECKING, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from .telemetry import extract_retriever_telemetry
from components.ingestor.ingestor import process_document

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
    ingestor_context = state.get("ingestor_context")

    # If we have ingestor_context, prepend it to raw_docs as a Document
    if ingestor_context:
        ingestor_doc = Document(
            page_content=ingestor_context,
            metadata={"source": "uploaded_file", "filename": state.get("filename", "unknown")}
        )
        raw_docs = [ingestor_doc] + raw_docs
        logger.info(f"Including ingestor context ({len(ingestor_context)} chars) with retrieved docs")

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

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Generation node failed: {e}", exc_info=True)
        metadata.update({
            "generation_duration": duration,
            "generation_success": False,
            "generation_error": str(e)
        })
        writer({"event": "error", "data": {"error": str(e)}})


async def ingest_node(state: 'GraphState') -> 'GraphState':
    """
    Node to process uploaded documents (PDF, DOCX) and extract chunked context.
    Only runs if file_content and filename are present in state.
    """
    start_time = datetime.now()

    file_content = state.get("file_content")
    filename = state.get("filename")
    metadata = state.get("metadata", {})

    # Skip if no file uploaded
    if not file_content or not filename:
        logger.info("No file to ingest, skipping ingest_node")
        return {}

    logger.info(f"Ingesting document: {filename}")

    try:
        # Process document and get chunked context
        ingestor_context = process_document(file_content, filename)

        duration = (datetime.now() - start_time).total_seconds()

        metadata.update({
            "ingest_duration": duration,
            "ingest_success": True,
            "ingested_filename": filename,
            "ingestor_context_length": len(ingestor_context)
        })

        logger.info(f"Document ingested successfully in {duration:.2f}s")

        return {
            "ingestor_context": ingestor_context,
            "metadata": metadata
        }

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)

        metadata.update({
            "ingest_duration": duration,
            "ingest_success": False,
            "ingest_error": str(e)
        })

        return {"ingestor_context": "", "metadata": metadata}


def _parse_filter_response(
    raw_response: str,
    filterable_fields: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate an LLM filter extraction response.
    Returns a validated {field: cast_value} dict, or None if parsing fails or result is empty.
    """
    try:
        cleaned = raw_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        extracted = json.loads(cleaned)
        if not isinstance(extracted, dict):
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    filters = {}
    for key, value in extracted.items():
        if key not in filterable_fields:
            continue
        declared_type = filterable_fields[key]
        try:
            if declared_type == "list":
                filters[key] = [str(v) for v in value] if isinstance(value, list) else [str(value)]
            elif declared_type == "int":
                filters[key] = int(value)
            else:
                filters[key] = str(value)
        except (TypeError, ValueError):
            pass  # drop uncastable values silently

    return filters if filters else None


async def extract_filters_node(
    state: "GraphState",
    generator: "Generator",
    filterable_fields: Dict[str, str],
    filter_values: Dict[str, list],
) -> "GraphState":
    """
    Node to extract metadata filters from query + user conversation history before retrieval.
    No-op if filterable_fields is empty. Fails gracefully — retrieval proceeds unfiltered on any error.

    Two-pass strategy:
    - Pass 1: LLM extracts from current query only (precise, no history noise)
    - Pass 2: if pass 1 yields nothing, LLM scans user_messages_history (user turns only,
              no assistant responses or retrieved document content) for filters established
              in prior turns and carries them forward
    """
    if not filterable_fields:
        logger.info("extract_filters_node: no filterable_fields configured, skipping")
        return {}  # No state update — metadata_filters stays None

    query = state.get("query", "")
    user_messages_history = state.get("user_messages_history")

    # Build field descriptions for the prompts, including valid values where available
    field_descriptions = []
    for field, ftype in filterable_fields.items():
        valid_vals = filter_values.get(field)
        if ftype == "list":
            base = f'"{field}" (list of strings, use JSON array)'
        else:
            base = f'"{field}" ({ftype})'
        if valid_vals:
            base += (
                f" — valid values: {valid_vals}. "
                "Pick the closest match from this list even if the user's wording differs slightly "
                "(e.g. a plural, typo, or synonym). Do NOT use a value outside this list."
            )
        field_descriptions.append(base)
    fields_desc = "\n".join(f"  - {d}" for d in field_descriptions)

    base_system = (
        "You are a metadata filter extraction assistant.\n"
        f"Available filterable fields:\n{fields_desc}\n\n"
        "Rules:\n"
        "- Only extract filters EXPLICITLY stated by the user. Do NOT infer or assume values.\n"
        "- For fields with a valid values list, always pick the closest match from that list.\n"
        "- For list-type fields, output a JSON array of strings.\n"
        "- For str/int fields, output a single value.\n"
        "- Return ONLY a valid JSON object, no markdown fences, no explanation.\n"
        "- If no filters are explicitly stated, return: {}\n"
        f"- Only use keys from: {list(filterable_fields.keys())}"
    )

    # --- Pass 1: current query only ---
    try:
        messages_p1 = [
            SystemMessage(content=base_system),
            HumanMessage(content=f"### CURRENT QUERY\n{query}\n\nExtract metadata filters as a JSON object."),
        ]
        raw_p1 = await generator._call_llm(messages_p1)
        filters = _parse_filter_response(raw_p1, filterable_fields)
    except Exception as e:
        logger.warning(f"extract_filters_node: pass 1 LLM call failed ({e}). Proceeding without filters.")
        return {"metadata_filters": None}

    if filters:
        logger.info(f"extract_filters_node: pass 1 extracted filters: {filters}")
        return {"metadata_filters": filters}

    # --- Pass 2: user history fallback ---
    if not user_messages_history:
        logger.info("extract_filters_node: no filters in current query and no user history, skipping")
        return {"metadata_filters": None}

    logger.info("extract_filters_node: pass 1 yielded nothing, checking user history")
    try:
        messages_p2 = [
            SystemMessage(content=base_system),
            HumanMessage(content=(
                f"### PREVIOUS USER MESSAGES\n{user_messages_history}\n\n"
                f"### CURRENT QUERY\n{query}\n\n"
                "The current query may be a follow-up. Extract any metadata filters that were "
                "established in the previous messages and still apply. "
                "Return {} if no filters were previously established."
            )),
        ]
        raw_p2 = await generator._call_llm(messages_p2)
        filters = _parse_filter_response(raw_p2, filterable_fields)
    except Exception as e:
        logger.warning(f"extract_filters_node: pass 2 LLM call failed ({e}). Proceeding without filters.")
        return {"metadata_filters": None}

    if filters:
        logger.info(f"extract_filters_node: pass 2 carried forward filters: {filters}")
    else:
        logger.info("extract_filters_node: no filters established in user history")

    return {"metadata_filters": filters}


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