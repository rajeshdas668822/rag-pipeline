"""FastAPI backend for RAG GUI system.

Exposes REST endpoints for health checks, model listing, query execution,
and batch evaluation.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    BatchQueryRequest,
    BatchQueryResponse,
    HealthResponse,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
)
from .rag_service import RAGService

# Data directory path
DATA_DIR = Path(__file__).parent.parent / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup: Initialize RAG service
    app.state.rag_service = RAGService(data_dir=DATA_DIR)
    yield
    # Shutdown: Cleanup if needed
    pass


app = FastAPI(
    title="RAG Pipeline API",
    description="Backend API for the RAG reranking pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for GUI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to GUI origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """List available Ollama models."""
    rag_service = app.state.rag_service
    models = await rag_service.list_models()
    return ModelsResponse(models=models)


@app.post("/api/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """Execute a RAG query with configuration."""
    rag_service = app.state.rag_service
    result = await rag_service.query(
        query=request.query,
        retriever_type=request.retriever_type,
        top_k=request.top_k,
        use_reranking=request.use_reranking,
        rerank_top_k=request.rerank_top_k,
        stream=request.stream,
    )
    return result


@app.post("/api/batch", response_model=BatchQueryResponse)
async def execute_batch(request: BatchQueryRequest):
    """Execute batch queries for evaluation."""
    rag_service = app.state.rag_service
    results = await rag_service.batch_query(
        queries=request.queries,
        retriever_type=request.retriever_type,
        top_k=request.top_k,
        use_reranking=request.use_reranking,
        rerank_top_k=request.rerank_top_k,
    )
    return results


@app.get("/api/evaluation", response_model=list)
async def load_evaluation_queries():
    """Load evaluation queries from data/evaluation_queries.jsonl."""
    evaluation_path = DATA_DIR / "evaluation_queries.jsonl"
    if not evaluation_path.exists():
        return []
    
    from src.evaluation_dataset import load_evaluation_queries as load_eval
    
    try:
        queries = load_eval(str(evaluation_path))
        return [
            {
                "query_id": q.query_id,
                "query": q.query,
                "relevant_documents": q.relevant_documents,
                "graded_labels": q.graded_labels,
            }
            for q in queries
        ]
    except Exception:
        return []
