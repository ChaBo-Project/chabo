from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel
from langchain_core.documents import Document

class GraphState(TypedDict):
    """State object passed through LangGraph workflow"""
    query: str
    context: str
    raw_context: list[Document]
    ingestor_context: str
    result: str
    sources: List[Dict[str, str]] 
    metadata: [Dict[str, Any]]
    file_content: Optional[bytes]
    filename: Optional[str]
    file_type: Optional[str]
    workflow_type: Optional[str]  # 'standard' or 'geojson_direct'
    metadata_filters: Optional[Dict[str, Any]]

class Message(BaseModel):
    """Single message in conversation history"""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    id: Optional[str] = None

class ChatUIInput(BaseModel):
    """Input model for text-only ChatUI requests"""
    messages: Optional[List[Message]] = None  # Structured conversation history
    preprompt: Optional[str] = None

class ChatUIFileInput(BaseModel):
    """Input model for ChatUI requests with file attachments"""
    files: Optional[List[Dict[str, Any]]] = None
    messages: Optional[List[Message]] = None  # Structured conversation history
    preprompt: Optional[str] = None
