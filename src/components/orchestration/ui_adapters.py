"""
ChatUI Adapters for LangGraph Workflow Streaming
"""
import logging
import asyncio
from typing import AsyncGenerator, Dict, Any

from components.utils import build_conversation_context

logger = logging.getLogger(__name__)


async def process_query_streaming(
    compiled_graph,
    query: str,
    file_upload=None,
    reports_filter: str = "",
    sources_filter: str = "",
    subtype_filter: str = "",
    year_filter: str = "",
    conversation_context: str = None
):
    """
    Process a query through the LangGraph workflow with streaming.

    COPIED FROM ORIGINAL ORCHESTRATOR. TO BE REPLACED WITH AGENTIC WORFLOW
    """
    # Build metadata filters from filter parameters
    metadata_filters = None
    if any([reports_filter, sources_filter, subtype_filter, year_filter]):
        metadata_filters = {}
        if reports_filter:
            metadata_filters["reports"] = reports_filter
        if sources_filter:
            metadata_filters["sources"] = sources_filter
        if subtype_filter:
            metadata_filters["subtype"] = subtype_filter
        if year_filter:
            metadata_filters["year"] = year_filter

    initial_state = {
        "query": query,
        "metadata": {"session_type": "chatui"},
        "raw_documents": [],
        "conversation_context": conversation_context,
        "metadata_filters": metadata_filters
    }

    try:
        async for output in compiled_graph.astream(initial_state, stream_mode="custom"):
            if output.get("event") == "data":
                yield {"type": "data", "content": output["data"]}
            elif output.get("event") == "sources":
                sources = output["data"].get("sources", [])
                yield {"type": "sources", "content": sources}
            elif output.get("event") == "error":
                yield {"type": "error", "content": output["data"].get("error", "Unknown error")}

        yield {"type": "end", "content": ""}

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield {"type": "error", "content": str(e)}


async def chatui_adapter(data, compiled_graph, max_turns: int = 3, max_chars: int = 8000):
    """Text-only adapter for ChatUI with structured message support"""
    logger.debug(f"ChatUI adapter called with data type: {type(data)}")

    try:
        # Handle both dict and object access patterns
        if isinstance(data, dict):
            text_value = data.get('text', '')
            messages_value = data.get('messages', None)
            preprompt_value = data.get('preprompt', None)
        else:
            text_value = getattr(data, 'text', '')
            messages_value = getattr(data, 'messages', None)
            preprompt_value = getattr(data, 'preprompt', None)

        # Convert dict messages to objects if needed
        messages = []
        if messages_value:
            for msg in messages_value:
                if isinstance(msg, dict):
                    messages.append(type('Message', (), {
                        'role': msg.get('role', 'unknown'),
                        'content': msg.get('content', '')
                    })())
                else:
                    messages.append(msg)

        # Extract latest user query
        user_messages = [msg for msg in messages if msg.role == 'user']
        query = user_messages[-1].content if user_messages else text_value

        # Conversation metadata (troubleshooting purposes)
        msg_metadata = {
            'total': len(messages),
            'user': len(user_messages),
            'assistant': len([m for m in messages if m.role == 'assistant']),
            'msg_lengths': [len(m.content) for m in messages]
        }
        logger.info(f"Processing query: {query[:20]}... | Conversation: {msg_metadata}")

        # Build conversation context for generation (last N turns)
        conversation_context = build_conversation_context(messages, max_turns=max_turns, max_chars=max_chars)
        full_response = ""
        sources_collected = None

        async for result in process_query_streaming(
            compiled_graph=compiled_graph,
            query=query,
            file_upload=None,
            reports_filter="",
            sources_filter="",
            subtype_filter="",
            year_filter="",
            conversation_context=conversation_context
        ):
            if isinstance(result, dict):
                result_type = result.get("type", "data")
                content = result.get("content", "")

                if result_type == "data":
                    full_response += content
                    yield content
                elif result_type == "sources":
                    sources_collected = content
                elif result_type == "end":
                    if sources_collected:
                        sources_text = "\n\n**Sources:**\n"
                        for i, source in enumerate(sources_collected, 1):
                            title = source.get('title', 'Unknown')
                            link = source.get('link') or '#'  # Handle empty string
                            sources_text += f"{i}. [{title}]({link})\n"
                        yield sources_text
                elif result_type == "error":
                    yield f"Error: {content}"
            else:
                yield str(result)

            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"ChatUI error: {str(e)}")
        logger.error(f"Full traceback:", exc_info=True)
        yield f"Error: {str(e)}"


async def chatui_file_adapter(data, compiled_graph, max_turns: int = 3, max_chars: int = 8000):
    """File upload adapter for ChatUI with structured message support"""
    try:
        # Handle both dict and object access patterns
        if isinstance(data, dict):
            text_value = data.get('text', '')
            messages_value = data.get('messages', None)
            files_value = data.get('files', None)
            preprompt_value = data.get('preprompt', None)
        else:
            text_value = getattr(data, 'text', '')
            messages_value = getattr(data, 'messages', None)
            files_value = getattr(data, 'files', None)
            preprompt_value = getattr(data, 'preprompt', None)

        # Extract query - prefer structured messages
        conversation_context = None
        if messages_value and len(messages_value) > 0:
            # Convert dict messages to objects
            messages = []
            for msg in messages_value:
                if isinstance(msg, dict):
                    messages.append(type('Message', (), {
                        'role': msg.get('role', 'unknown'),
                        'content': msg.get('content', '')
                    })())
                else:
                    messages.append(msg)

            user_messages = [msg for msg in messages if msg.role == 'user']
            query = user_messages[-1].content if user_messages else text_value

            # Conversation metadata (troubleshooting purposes)
            msg_metadata = {
                'total': len(messages),
                'user': len(user_messages),
                'assistant': len([m for m in messages if m.role == 'assistant']),
                'msg_lengths': [len(m.content) for m in messages]
            }
            logger.info(f"Processing query with file: {query[:20]}... | Conversation: {msg_metadata}")

            conversation_context = build_conversation_context(messages, max_turns=max_turns, max_chars=max_chars)
        else:
            query = text_value

        file_content = None
        filename = None

        if files_value and len(files_value) > 0:
            file_info = files_value[0]
            logger.info(f"Processing file: {file_info.get('name', 'unknown')}")

            if file_info.get('type') == 'base64' and file_info.get('content'):
                try:
                    import base64
                    file_content = base64.b64decode(file_info['content'])
                    filename = file_info.get('name', 'uploaded_file')
                except Exception as e:
                    logger.error(f"Error decoding base64 file: {str(e)}")
                    yield f"Error: Failed to decode uploaded file - {str(e)}"
                    return

        sources_collected = None

        async for result in process_query_streaming(
            compiled_graph=compiled_graph,
            query=query,
            file_upload=None,  # File handling can be extended
            reports_filter="",
            sources_filter="",
            subtype_filter="",
            year_filter="",
            conversation_context=conversation_context
        ):
            if isinstance(result, dict):
                result_type = result.get("type", "data")
                content = result.get("content", "")

                if result_type == "data":
                    yield content
                elif result_type == "sources":
                    sources_collected = content
                elif result_type == "end":
                    if sources_collected:
                        sources_text = "\n\n**Sources:**\n"
                        for i, source in enumerate(sources_collected, 1):
                            if isinstance(source, dict):
                                title = source.get('title', 'Unknown')
                                link = source.get('link') or '#'  # Handle empty string
                                sources_text += f"{i}. [{title}]({link})\n"
                            else:
                                sources_text += f"{i}. {str(source)}\n"
                        yield sources_text
                elif result_type == "error":
                    yield f"Error: {content}"
            else:
                yield str(result)

            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"ChatUI file adapter error: {str(e)}")
        yield f"Error: {str(e)}"