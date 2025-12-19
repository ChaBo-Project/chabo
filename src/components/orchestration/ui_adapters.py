import logging
from typing import AsyncGenerator, Dict, Any
import json

logger = logging.getLogger(__name__)

async def chatui_stream_adapter(payload: Dict[str,Any], graph_instance: Any) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Runs the graph and yields events in ChatUI format: 
    {'type': 'data', 'content': '...'}
    """
    
    input_data = payload.get("input", {})
    query = input_data.get("text")
    
    if not query and input_data.get("messages"):
        query = input_data["messages"][-1].get("content")

    if not query:
        yield json.dumps({"type": "error", "content": "No query found in payload"})
        return

    initial_state = {
        "query": query,
        "metadata": {"session_type": "chatui"},
        "raw_documents": []
    }

    try:
        # We use astream with 'custom' to capture the yields from generate_node
        async for output in graph_instance.astream(initial_state, stream_mode="custom"):
            
            # 1. Handle Token Chunks
            if output.get("event") == "data":
                yield json.dumps({"type": "data", "content": output["data"]}) + "\n"
            
            # 2. Handle Final Sources
            elif output.get("event") == "sources":
                # Formats sources into a nice string for the ChatUI
                sources = output["data"].get("sources", [])
                source_text = "\n\n**Sources:**\n" + "\n".join(
                    [f"{i+1}. [{s['title']}]({s.get('link', '#')})" for i, s in enumerate(sources)]
                )
                yield json.dumps({"type": "sources", "content": source_text})

        # 3. Final Signal
        yield json.dumps({"type": "end", "content": ""})

    except Exception as e:
        logger.error(f"Adapter error: {e}")
        yield json.dumps({"type": "error", "content": str(e)})