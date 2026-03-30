import os
import json
import configparser
import logging
import asyncio
logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Union, Optional
from pydantic import Field
from qdrant_client import QdrantClient, AsyncQdrantClient
from gradio_client import Client as GradioClient
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..utils import getconfig, get_config_value, _call_hf_endpoint, _acall_hf_endpoint
from qdrant_client.http import models as rest


def _build_qdrant_filter(filters: Dict[str, Any]) -> Optional[rest.Filter]:
    """
    Convert a plain {field: value} dict into a Qdrant rest.Filter.
    Payload fields are nested under 'metadata', so keys become 'metadata.<field>'.
    - list value   → MatchAny  (match any element in list)
    - scalar value → MatchValue (exact match)
    All conditions ANDed via must[]. Returns None if filters is empty or None.
    """
    if not filters:
        return None

    must_conditions = []
    for field, value in filters.items():
        qdrant_key = f"metadata.{field}"
        if isinstance(value, list):
            must_conditions.append(
                rest.FieldCondition(key=qdrant_key, match=rest.MatchAny(any=value))
            )
        else:
            must_conditions.append(
                rest.FieldCondition(key=qdrant_key, match=rest.MatchValue(value=value))
            )

    return rest.Filter(must=must_conditions)


def _format_hit(hit) -> Dict[str, Any]:
    """Format a Qdrant ScoredPoint into the standard retriever result dict."""
    return {
        "answer": hit.payload.get("text", hit.payload.get("page_content", "")),
        "answer_metadata": hit.payload.get("metadata", {}),
        "score": hit.score
    }


# --- THE MAIN RETRIEVER ORCHESTRATOR CLASS  ---
class ChaBoHFEndpointRetriever(BaseRetriever):
    """
    LangChain Retriever that orchestrates three decoupled microservices:
    1. HF Endpoint for Embedding.
    2. Dynamic Qdrant search (Native or Gradio Client).
    3. HF Endpoint for Reranking.
    """
    # Configuration Fields (Used for pydantic validation and internal storage)
    hf_token: str
    embedding_endpoint_url: str
    reranker_endpoint_url: str
    
    qdrant_mode: str
    qdrant_url: str
    qdrant_api_key: Union[str, None]
    qdrant_port: int
    qdrant_collection: str
    
    initial_k: int
    final_k: int

    # We use separate caches for the sync and async clients
    sync_qdrant_client: QdrantClient = Field(default=None, exclude=True)
    async_qdrant_client: AsyncQdrantClient = Field(default=None, exclude=True)
    gradio_client: GradioClient = Field(default=None, exclude=True)


    # --- Client Lazy Initialization  ---
    @classmethod
    def from_config(cls, **kwargs) -> 'ChaBoHFEndpointRetriever':
        """Initializes the class and the appropriate Qdrant client based on qdrant_mode."""
        instance = cls(**kwargs)
        
        mode = instance.qdrant_mode.lower()
        if mode not in ['native', 'gradio']:
            logger.error(f"Unsupported qdrant_mode: {mode}. Must be 'native' or 'gradio'.")
            raise ValueError(f"Unsupported qdrant_mode: {mode}. Must be 'native' or 'gradio'.")
        
        # No client initialization here!
        logger.info(f"Retriever initialized for {mode} mode (Clients will be loaded lazily).")
        return instance

    # ---  Client Init (Synchronous) ---
    def _get_qdrant_client(self) -> Union[QdrantClient, GradioClient]:
        """Returns the appropriate synchronous client, initializing it if necessary."""
        if self.qdrant_mode.lower() == 'native':
            if not self.sync_qdrant_client:
                logger.info(f"Lazy Init: Creating Sync Native QdrantClient from  {self.qdrant_url}")
                self.sync_qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    port=self.qdrant_port,
                    https=True, 
                    api_key=self.qdrant_api_key or self.hf_token, 
                    timeout=60
                )
            return self.sync_qdrant_client
        
        elif self.qdrant_mode.lower() == 'gradio':
            if not self.gradio_client:
                logger.info(f"Lazy Init: Creating GradioClient from {self.qdrant_url}")
                self.gradio_client = GradioClient(
                    self.qdrant_url, 
                    hf_token=self.qdrant_api_key # Assuming Gradio uses HF_TOKEN
                )
            return self.gradio_client
        
        raise ValueError("Invalid qdrant_mode.")

    # --- Client Init (Asynchronous) ---
    async def _aget_qdrant_client(self)->Union[AsyncQdrantClient, GradioClient]:
        """Returns the appropriate asynchronous client, initializing it if necessary."""
        if self.qdrant_mode.lower() == 'native':
            if not self.async_qdrant_client:
                logger.info(f"Lazy Init: Creating Async Native QdrantClient from {self.qdrant_url}")
                self.async_qdrant_client = AsyncQdrantClient(
                    url=self.qdrant_url, 
                    port=self.qdrant_port,
                    api_key=self.qdrant_api_key or self.hf_token, 
                    timeout=60
                )
            return self.async_qdrant_client
        
        # Gradio client object is synchronous but its predict method is inherently async
        return self._get_qdrant_client()
    
    # --- Qdrant Synchronous Search Helper (Handles Mode Switching) ---
    def _search_qdrant(self, query_vector: List[float], filters: Dict = None) -> List[Dict[str, Any]]:
        """Performs the synchronous Qdrant search. If mode is gradio expects
         the api_endpoint = 'query_points' similar to native mode"""
        
        try:

            client = self._get_qdrant_client()
            
            if self.qdrant_mode.lower() == 'native':
                logger.debug(f"Sync Native Qdrant search: collection={self.qdrant_collection}, k={self.initial_k}")
                search_result = client.query_points(
                    collection_name=self.qdrant_collection,
                    query=query_vector,
                    query_filter=_build_qdrant_filter(filters),
                    limit=self.initial_k,
                    with_payload=True,
                    with_vectors=False
                )

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not search_result.points and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results, retrying with priority field only: {priority_filter}"
                    )
                    search_result = client.query_points(
                        collection_name=self.qdrant_collection,
                        query=query_vector,
                        query_filter=_build_qdrant_filter(priority_filter),
                        limit=self.initial_k,
                        with_payload=True,
                        with_vectors=False
                    )

                return [_format_hit(hit) for hit in search_result.points]

            elif self.qdrant_mode.lower() == 'gradio':
                logger.debug(f"Sync Gradio Qdrant search: collection={self.qdrant_collection}, k={self.initial_k}")
                result = client.predict(
                    query_vector_json=json.dumps(query_vector),
                    collection_name=self.qdrant_collection,
                    top_k=self.initial_k,
                    query_filter=json.dumps(filters) if filters else None,
                    api_name="/query_points"
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Gradio wrapper error: {result.get('message', result)}")
                    return []

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not result and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results (Gradio), retrying with priority field only: {priority_filter}"
                    )
                    result = client.predict(
                        query_vector_json=json.dumps(query_vector),
                        collection_name=self.qdrant_collection,
                        top_k=self.initial_k,
                        query_filter=json.dumps(priority_filter),
                        api_name="/query_points"
                    )
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Gradio wrapper error on priority retry: {result.get('message', result)}")
                        return []

                return result
        
        except Exception as e:
            logger.error(f"Search failed at {self.qdrant_url}. Error: {e}")
            return []
    
    # --- Qdrant Asynchronous search ---
    async def _asearch_qdrant(self, query_vector: List[float], filters: Dict = None) -> List[Dict[str, Any]]:
        """Performs the asynchronous Qdrant search. If mode is gradio expects
         the api_endpoint = 'query_points' similar to native mode"""
        try:
            client = await self._aget_qdrant_client()

            if self.qdrant_mode.lower() == 'native':
                logger.debug(f"Async Native Qdrant search: collection={self.qdrant_collection}, k={self.initial_k}")

                search_result = await client.query_points(
                    collection_name=self.qdrant_collection,
                    query=query_vector,
                    query_filter=_build_qdrant_filter(filters),
                    limit=self.initial_k,
                    with_payload=True,
                    with_vectors=False
                )

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not search_result.points and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results, retrying with priority field only: {priority_filter}"
                    )
                    search_result = await client.query_points(
                        collection_name=self.qdrant_collection,
                        query=query_vector,
                        query_filter=_build_qdrant_filter(priority_filter),
                        limit=self.initial_k,
                        with_payload=True,
                        with_vectors=False
                    )

                return [_format_hit(hit) for hit in search_result.points]

            elif self.qdrant_mode.lower() == 'gradio':
                logger.debug(f"Async Gradio Qdrant search: collection={self.qdrant_collection}, k={self.initial_k}")
                loop = asyncio.get_running_loop()

                # Use run_in_executor to make the synchronous .predict() awaitable
                result = await loop.run_in_executor(
                    None,
                    lambda: client.predict(
                        query_vector_json=json.dumps(query_vector),
                        collection_name=self.qdrant_collection,
                        top_k=self.initial_k,
                        query_filter=json.dumps(filters) if filters else None,
                        api_name="/query_points"
                    )
                )
                if isinstance(result, dict) and "error" in result:
                    logger.error(f"Gradio wrapper error: {result.get('message', result)}")
                    return []

                # Safeguard: if AND filter returns 0 results and multiple fields were applied,
                # retry with the priority field only (first key in filters dict).
                if not result and filters and len(filters) > 1:
                    priority_field = next(iter(filters))
                    priority_filter = {priority_field: filters[priority_field]}
                    logger.info(
                        f"AND filter returned 0 results (Gradio), retrying with priority field only: {priority_filter}"
                    )
                    result = await loop.run_in_executor(
                        None,
                        lambda: client.predict(
                            query_vector_json=json.dumps(query_vector),
                            collection_name=self.qdrant_collection,
                            top_k=self.initial_k,
                            query_filter=json.dumps(priority_filter),
                            api_name="/query_points"
                        )
                    )
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"Gradio wrapper error on priority retry: {result.get('message', result)}")
                        return []

                return result
        except Exception as e:
            logger.error(f"Search failed at {self.qdrant_url}. Error: {e}")
            return []
        
        
    # --- Core Retrieval Orchestration (LangChain Required Method) ---
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Executes the three-step pipeline: Embed -> Search -> Rerank.
        """
        # A. Embed Query (Call HF Endpoint 1)
        logger.info(f"Emebedding query: {query[:50]}....")
        try:

            embed_payload = {"inputs": query}
            embed_response = _call_hf_endpoint(
                self.embedding_endpoint_url, 
                self.hf_token, 
                embed_payload
            )
            query_vector = embed_response[0] 
        except Exception as e:
            logger.error(f"CRITICAL: Embedding Failed. Details: {e}")
            return []
        
                
        # B. Search Qdrant (Dynamic Call)
        candidate_results = self._search_qdrant(query_vector, filters=kwargs.get("filters"))
        logger.debug(f"Candidate Results {candidate_results}")
        if not candidate_results:
            logger.info(f"No candidates found for query: {query[:50]}...")
            return []

        # C. Rerank Documents (Call HF Endpoint 2), with fallback in case of error
        documents = []
        try:
            reranker_payload = {
                "query": query,
                "texts": [candidate["answer"] for candidate in candidate_results]
            }
            logger.info(f"Performing Reranking for {len(candidate_results)}")
            final_reranked_results = _call_hf_endpoint(
                self.reranker_endpoint_url, 
                self.hf_token, 
                reranker_payload)
        
            # D. Format and Return
            
            for doc_data in final_reranked_results[:self.final_k]: 
                # 1. Get the original index and the new score
                original_index = doc_data['index']
                rerank_score = doc_data['score']
        
                # 2. Retrieve the original document data using the index
                original_doc_data = candidate_results[original_index]

                # 3. Extract content and metadata
                content = original_doc_data.get("answer", original_doc_data.get("page_content", ""))
                metadata = original_doc_data.get("answer_metadata", original_doc_data.get("metadata", {})).copy()
                metadata['retriever_score'] = original_doc_data.get("score")
                metadata['rerank_score'] = doc_data.get('score')
                
                documents.append(
                    Document(page_content=content, metadata=metadata)
                )
        except Exception as e:
            # FALLBACK: If Reranker fails (503/Timeout), return top candidates from vector search
            logger.warning(f"NON-CRITICAL: Reranking failed ({e}). Falling back to vector search order.")
            for original_doc_data in candidate_results[:self.final_k]:
                content = original_doc_data.get("answer", "")
                metadata = original_doc_data.get("answer_metadata", original_doc_data.get("metadata", {})).copy()
                metadata['retriever_score'] = original_doc_data.get("score")
                metadata['rerank_score'] = "FALLBACK" # Indicate fallback in metadata
                documents.append(Document(page_content=content, metadata=metadata))
                  
        return documents
    

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        [ASYNC METHOD IMPLEMENTATION] Executes the three-step pipeline: Embed -> Search -> Rerank.
        """
        # A. Embed Query (Call HF Endpoint 1)
        logger.info(f"Emebedding query: {query[:50]}....")
        try:
            embed_payload = {"inputs": query}
            embed_response = await _acall_hf_endpoint( 
                self.embedding_endpoint_url, 
                self.hf_token, 
                embed_payload
            )
            query_vector = embed_response[0]
        except Exception as e:
            logger.error(f"CRITICAL: Embedding Failed. Details: {e}")
            return []


        # B. Search Qdrant (Dynamic Async Call)
        candidate_results = await self._asearch_qdrant(query_vector, filters=kwargs.get("filters")) 
        logger.debug(f"Candidate Results {candidate_results}")

        if not candidate_results:
            logger.info(f"No candidates found for query: {query[:50]}...")
            return []
        
        # C. Rerank Documents (Call HF Endpoint 2)

        documents = []
        try:
            reranker_payload = {
                "query": query,
                "texts": [candidate["answer"] for candidate in candidate_results]
            }
            logger.info(f"Async Reranking for {len(candidate_results)} candidates")
        
            final_reranked_results = await _acall_hf_endpoint(
                self.reranker_endpoint_url, 
                self.hf_token, 
                reranker_payload
            )

            # D. Format and Return
            documents = []
            for doc_data in final_reranked_results[:self.final_k]: 
                # 1. Get the original index and the new score
                original_index = doc_data['index']
                rerank_score = doc_data['score']
        
                # 2. Retrieve the original document data using the index
                original_doc_data = candidate_results[original_index]

                # 3. Extract content and metadata
                content = original_doc_data.get("answer", original_doc_data.get("page_content", ""))
                metadata = original_doc_data.get("answer_metadata", original_doc_data.get("metadata", {})).copy()
                metadata['retriever_score'] = original_doc_data.get("score")
                metadata['rerank_score'] = doc_data.get('score')
                
                documents.append(
                    Document(page_content=content, metadata=metadata)
                )
        except Exception as e:
        # FALLBACK: Return top results from initial vector search
            logger.warning(f"NON-CRITICAL: Async Reranking failed ({e}). Returning search results.")
            for original_doc_data in candidate_results[:self.final_k]:
                metadata = original_doc_data.get("answer_metadata", {}).copy()
                metadata['retriever_score'] = original_doc_data.get("score")
                metadata['rerank_score'] = "FALLBACK"
                
                documents.append(Document(
                    page_content=original_doc_data.get("answer", ""), 
                    metadata=metadata
                ))
            
        return documents
    

def create_retriever_from_config(config_file: str = "params.cfg"):
    """Loads configuration and instantiates the CustomHFRAGRetriever."""
    config = getconfig(config_file)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required but not set")

    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable is required but not set")

    config_map = {
        "embedding_endpoint_url": ("hf_endpoints", "embedding_endpoint_url", "EMBEDDING_ENDPOINT_URL"),
        "reranker_endpoint_url":  ("hf_endpoints", "reranker_endpoint_url", "RERANKER_ENDPOINT_URL"),

        "qdrant_mode":          ("qdrant", "mode", "QDRANT_MODE"),
        "qdrant_url":           ("qdrant", "url", "QDRANT_URL"),
        "qdrant_port":          ("qdrant", "port", "QDRANT_PORT"),
        "qdrant_collection":    ("qdrant", "collection", "QDRANT_COLLECTION"),

        "initial_k":            ("retrieval", "initial_k", "RETRIEVAL_INITIAL_K", 20),
        "final_k":              ("retrieval", "final_k", "RETRIEVAL_FINAL_K", 5),
    }

    retriever_config_kwargs = {
        "hf_token": hf_token,
        "qdrant_api_key": qdrant_api_key
    }

    for key, params in config_map.items():

        section, option, env_var = params[:3]
        fallback = params[3] if len(params) > 3 else None
        value = get_config_value(config, section, option, env_var, fallback)
        
        if key in ['initial_k', 'top_k', 'qdrant_port']:
            value = int(value)

            
        retriever_config_kwargs[key] = value

    logger.info(f"Configuration loaded. Qdrant Mode: {retriever_config_kwargs['qdrant_mode']}")
    return ChaBoHFEndpointRetriever.from_config(**retriever_config_kwargs)