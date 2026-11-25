"""Pydantic models for RAG API requests and responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"


class ModelsResponse(BaseModel):
    """Response for available models list."""
    models: list[str]


class QueryRequest(BaseModel):
    """Request for single query endpoint."""
    query: str
    retriever_type: str = "dense"  # "dense", "bm25", "hybrid"
    top_k: int = 50
    use_reranking: bool = False
    rerank_top_k: Optional[int] = None
    stream: bool = False


class QueryResponse(BaseModel):
    """Response for single query endpoint."""
    query_id: str
    query: str
    answer: str
    citations: list[str]
    insufficient_context: bool
    retrieval_results: list[dict]
    reranked_results: Optional[list[dict]] = None
    generation_ms: float
    retrieval_ms: float
    reranking_ms: Optional[float] = None


class BatchQueryRequest(BaseModel):
    """Request for batch query endpoint."""
    queries: list[str]
    query_ids: Optional[list[str]] = None
    retriever_type: str = "dense"
    top_k: int = 50
    use_reranking: bool = False
    rerank_top_k: Optional[int] = None


class BatchQueryResponse(BaseModel):
    """Response for batch query endpoint."""
    results: list[dict]
    total_queries: int
    successful: int
    failed: int


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    doc_id: str
    score: float
    rank: int


class RerankResult(BaseModel):
    """Single reranking result."""
    doc_id: str
    rerank_score: float
    reranked_position: int
    initial_rank: int
    initial_score: float
