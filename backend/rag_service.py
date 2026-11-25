"""RAG orchestration service for FastAPI backend.

Uses existing RAG components from src/ module.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from src.bm25_retriever import BM25Retriever
from src.chunker import Chunker
from src.config import Config
from src.cross_encoder_reranker import CrossEncoderReranker, ModelLoadError
from src.dense_retriever import DenseRetriever, NumpyBackend
from src.document_loader import DocumentLoader
from src.embedding_cache import CachedEmbedder
from src.evaluation_dataset import load_evaluation_queries
from src.hybrid_retriever import HybridRetriever
from src.ollama_client import OllamaClient, OllamaModelMissingError, OllamaUnavailableError
from src.ollama_embedder import OllamaEmbedder
from src.ollama_generator import OllamaGenerator
from src.rag_types import Chunk


class RAGService:
    """RAG orchestration service using existing components."""

    def __init__(self, data_dir: Path):
        """Initialize RAG service.
        
        Args:
            data_dir: Path to data directory containing documents.
        """
        self.data_dir = data_dir
        self.config = Config.load()
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(endpoint=self.config.ollama_endpoint)
        
        # Lazy initialization flags
        self._initialized = False
        self._chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}
        self._dense: Optional[DenseRetriever] = None
        self._bm25: Optional[BM25Retriever] = None
        self._hybrid: Optional[HybridRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._generator: Optional[OllamaGenerator] = None

    async def _initialize(self) -> None:
        """Initialize RAG pipeline components."""
        if self._initialized:
            return
        
        # Ensure Ollama is available
        try:
            self.ollama_client.ensure_ready([
                self.config.embedding_model,
                self.config.instruct_model,
            ])
        except OllamaUnavailableError as e:
            raise RuntimeError(f"Ollama endpoint unavailable: {e}")
        except OllamaModelMissingError as e:
            raise RuntimeError(f"Required Ollama model missing: {e}")
        
        # Load documents
        documents = DocumentLoader(self.config).load_directory(str(self.data_dir))
        
        # Chunk documents
        chunker = Chunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self._chunks = []
        for doc in documents:
            self._chunks.extend(chunker.chunk(doc))
        
        self._chunk_map = {c.chunk_id: c for c in self._chunks}
        
        # Initialize embedder
        embedder = OllamaEmbedder(
            client=self.ollama_client,
            model=self.config.embedding_model,
            timeout_s=self.config.embed_timeout_s,
        )
        embedder = CachedEmbedder(embedder)
        
        # Initialize retrievers
        dense_backend = NumpyBackend()
        self._dense = DenseRetriever(embedder=embedder, backend=dense_backend)
        self._dense.index(self._chunks)
        
        self._bm25 = BM25Retriever()
        self._bm25.index(self._chunks)
        
        self._hybrid = HybridRetriever(
            dense=self._dense,
            bm25=self._bm25,
            rrf_k=self.config.rrf_k,
            fetch_k=50,
        )
        
        # Initialize reranker
        try:
            self._reranker = CrossEncoderReranker(
                model_name=self.config.cross_encoder_model,
            )
            self._reranker.index(self._chunk_map)
        except ModelLoadError as e:
            # Reranker is optional - log warning but continue
            print(f"Warning: Could not load cross-encoder reranker: {e}")
            self._reranker = None
        
        # Initialize generator
        self._generator = OllamaGenerator(
            client=self.ollama_client,
            model=self.config.instruct_model,
            timeout_s=self.config.generate_timeout_s,
        )
        
        self._initialized = True

    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            return self.ollama_client.available_models()
        except OllamaUnavailableError:
            return []

    async def query(
        self,
        query: str,
        retriever_type: str = "dense",
        top_k: int = 50,
        use_reranking: bool = False,
        rerank_top_k: Optional[int] = None,
        stream: bool = False,
    ) -> dict:
        """Execute a single RAG query.
        
        Args:
            query: User query string.
            retriever_type: "dense", "bm25", or "hybrid".
            top_k: Number of candidates to retrieve.
            use_reranking: Whether to apply cross-encoder reranking.
            rerank_top_k: Number of results after reranking (optional).
            stream: Whether to stream results (not implemented yet).
            
        Returns:
            Dictionary with query results.
        """
        await self._initialize()
        
        # Select retriever
        retriever_map = {
            "dense": self._dense,
            "bm25": self._bm25,
            "hybrid": self._hybrid,
        }
        retriever = retriever_map.get(retriever_type)
        if retriever is None:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        # Retrieve candidates
        retrieval_output = retriever.retrieve(query, top_k)
        retrieval_results = [
            {
                "doc_id": r.doc_id,
                "score": r.score,
                "rank": r.rank,
            }
            for r in retrieval_output.results
        ]
        
        # Rerank if enabled
        reranked_results = None
        reranking_ms = None
        
        if use_reranking and self._reranker is not None:
            candidates = [r for r in retrieval_output.results]
            rerank_k = rerank_top_k or top_k
            rerank_output = self._reranker.rerank(query, candidates, k=rerank_k)
            
            reranked_results = [
                {
                    "doc_id": r.doc_id,
                    "rerank_score": r.rerank_score,
                    "reranked_position": r.reranked_position,
                    "initial_rank": r.initial_rank,
                    "initial_score": r.initial_score,
                }
                for r in rerank_output.results
            ]
            reranking_ms = rerank_output.latency_ms
        
            # Get chunks from reranked results
            chunk_ids = [r.doc_id for r in rerank_output.results]
            chunks = [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]
        else:
            # Get chunks from retrieval results
            chunk_ids = [r.doc_id for r in retrieval_output.results]
            chunks = [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]
        
        # Generate answer
        gen_output = self._generator.generate(query, chunks)
        
        return {
            "query_id": "single_query",
            "query": query,
            "answer": gen_output.answer,
            "citations": gen_output.citations,
            "insufficient_context": gen_output.insufficient_context,
            "retrieval_results": retrieval_results,
            "reranked_results": reranked_results,
            "generation_ms": gen_output.latency_ms,
            "retrieval_ms": retrieval_output.latency_ms,
            "reranking_ms": reranking_ms,
        }

    async def batch_query(
        self,
        queries: list[str],
        query_ids: Optional[list[str]] = None,
        retriever_type: str = "dense",
        top_k: int = 50,
        use_reranking: bool = False,
        rerank_top_k: Optional[int] = None,
    ) -> dict:
        """Execute batch queries.
        
        Args:
            queries: List of query strings.
            query_ids: Optional list of query IDs (auto-generated if not provided).
            retriever_type: Retriever type to use.
            top_k: Number of candidates to retrieve.
            use_reranking: Whether to apply reranking.
            rerank_top_k: Number of results after reranking.
            
        Returns:
            Dictionary with batch results.
        """
        await self._initialize()
        
        if query_ids is None:
            query_ids = [f"batch_{i}" for i in range(len(queries))]
        
        if len(query_ids) != len(queries):
            raise ValueError("query_ids and queries must have same length")
        
        results = []
        successful = 0
        failed = 0
        
        for query_id, query in zip(query_ids, queries):
            try:
                result = await self.query(
                    query=query,
                    retriever_type=retriever_type,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    rerank_top_k=rerank_top_k,
                )
                result["query_id"] = query_id
                results.append(result)
                successful += 1
            except Exception as e:
                results.append({
                    "query_id": query_id,
                    "query": query,
                    "error": str(e),
                    "answer": None,
                    "citations": [],
                })
                failed += 1
        
        return {
            "results": results,
            "total_queries": len(queries),
            "successful": successful,
            "failed": failed,
        }
