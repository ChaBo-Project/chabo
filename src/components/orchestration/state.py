"""
Pydantic models for orchestration state and ChatUI input
"""
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_core.documents import Document

class GraphState(TypedDict, total=False):
    """State object passed through LangGraph workflow"""
    query: str
    context: str
    raw_documents: List[Document]  # Retrieved documents for generation
    raw_context: List[Document]    # Alias for backward compatibility
    ingestor_context: str
    result: str
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]
    conversation_context: Optional[str]  # Conversation history for multi-turn
    file_content: Optional[bytes]
    filename: Optional[str]
    file_type: Optional[str]
    workflow_type: Optional[str]  # 'standard' or 'geojson_direct'
    metadata_filters: Optional[Dict[str, Any]]
    user_messages_history: Optional[str]  # User-turn-only history for filter extraction (no assistant responses)
    applied_filters: Optional[Dict[str, Any]]  # Actual filters used after AND-safeguard fallback
    filters_narrowed: Optional[bool]  # True if AND safeguard fired and fell back to priority field

class Message(BaseModel):
    """Single message in conversation history"""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    id: Optional[str] = None

class ChatUIInput(BaseModel):
    """Input model for text-only ChatUI requests"""
    messages: Optional[List[Message]] = None
    preprompt: Optional[str] = None

class ChatUIFileInput(BaseModel):
    """Input model for ChatUI requests with file attachments"""
    files: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[Message]] = None
    preprompt: Optional[str] = None
