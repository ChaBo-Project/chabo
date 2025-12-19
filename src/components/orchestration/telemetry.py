# Helper function to extract configuration and score data from documents
import json
from typing import Dict, Any, List
from langchain_core.documents import Document


def extract_retriever_telemetry(docs: List[Document], retriever_config:Dict[str,Any]) -> Dict[str, Any]:
    """Extracts min/max scores and configuration metadata from the retrieved documents."""
    if not docs:
        return {
            "total_docs_retrieved": 0,
            "min_rerank_score": None,
            "max_rerank_score": None,
            "min_retriever_score": None,
            "max_retriever_score": None,
            "initial_k_config": retriever_config.get("initial_k"),
            "final_k_config": retriever_config.get("final_k"),
            }
    
    # Assuming 'rerank_score' and 'retriever_score' are added by your orchestrator
    rerank_scores = [doc.metadata.get('rerank_score') for doc in docs if doc.metadata.get('rerank_score') is not None]
    retriever_scores = [doc.metadata.get('retriever_score') for doc in docs if doc.metadata.get('retriever_score') is not None]

    telemetry = {
        "total_docs_retrieved": len(docs),
        "initial_k_config": retriever_config.get("initial_k"),
        "final_k_config": retriever_config.get("final_k"),
        "min_rerank_score": min(rerank_scores) if rerank_scores else None,
        "max_rerank_score": max(rerank_scores) if rerank_scores else None,
        "min_retriever_score": min(retriever_scores) if retriever_scores else None,
        "max_retriever_score": max(retriever_scores) if retriever_scores else None,
        # The true initial_k is often only known by the orchestrator, but we capture what we can
    }
    return telemetry