"""Dense vector retriever with NumPy backend.

This module provides a dense retriever implementation using cosine similarity
search. It satisfies Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 16.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import Config
from .ollama_embedder import OllamaEmbedder
from .rag_types import Chunk, RetrievalResult


class VectorBackend(Protocol):
    """Protocol for vector similarity backends."""

    def build(self, matrix: np.ndarray) -> None:
        """Build an index from the embedding matrix.

        Args:
            matrix: 2D numpy array of shape (n_vectors, dimension).
        """
        ...

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        """Search for top-k most similar vectors.

        Args:
            query_vec: 1D numpy array query vector.
            k: Number of results to return.

        Returns:
            Tuple of (indices, scores) for top-k results.
            Indices are sorted by descending score; ties broken by ascending index.
        """
        ...


class NumpyBackend(VectorBackend):
    """NumPy-based vector backend with from-scratch cosine similarity.

    No external similarity or ranking libraries are used.
    """

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._dimension: int | None = None

    def build(self, matrix: np.ndarray) -> None:
        """Build the index by storing the embedding matrix and precomputing norms.

        Args:
            matrix: 2D numpy array of shape (n_vectors, dimension).

        Raises:
            ValueError: If matrix has zero magnitude vectors or dimension mismatch
                for existing data.
        """
        # Handle empty matrix case
        if matrix.size == 0:
            self._matrix = matrix.reshape((0, self._dimension if hasattr(self, '_dimension') and self._dimension else 0))
            self._norms = np.array([])
            return

        # Store dimension from first build
        if not hasattr(self, '_dimension') or self._dimension is None:
            self._dimension = matrix.shape[1]

        # Compute L2 norms for each vector
        norms = np.linalg.norm(matrix, axis=1)

        # Check for zero-norm vectors
        if np.any(norms == 0):
            raise ValueError("Embedding vectors must have non-zero magnitude.")

        self._matrix = matrix
        self._norms = norms

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> tuple[list[int], list[float]]:
        """Search for top-k most similar vectors using cosine similarity.

        Args:
            query_vec: 1D numpy array query vector.
            k: Number of results to return.

        Returns:
            Tuple of (indices, scores) for top-k results.
            Indices are sorted by descending score; ties broken by ascending index.

        Raises:
            ValueError: If query vector has zero magnitude or dimension mismatch.
        """
        if self._matrix is None or self._norms is None:
            raise ValueError("Backend has not been built. Call build() first.")

        # Check dimension match
        if query_vec.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Query dimension {query_vec.shape[0]} does not match "
                f"expected {self._matrix.shape[1]}."
            )

        # Compute query norm
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            raise ValueError("Query vector must have non-zero magnitude.")

        # Compute cosine similarity: (query . matrix) / (||query|| * ||matrix||)
        # Using dot product and broadcasting
        dot_products = np.dot(self._matrix, query_vec)
        similarities = dot_products / (query_norm * self._norms)

        # Get indices and scores
        n_vectors = self._matrix.shape[0]

        # Handle k values
        if k < 1:
            return [], []

        # Cap k at number of vectors
        k = min(k, n_vectors)

        if k == 0:
            return [], []

        # Create array of indices [0, 1, 2, ..., n_vectors-1]
        indices = np.arange(n_vectors)

        # Sort by: primary = descending score, secondary = ascending index
        # We use negative scores for descending, then indices for ties
        sorted_indices = np.lexsort((indices, -similarities))

        # Take top k
        top_k_indices = sorted_indices[:k].tolist()
        top_k_scores = similarities[top_k_indices].tolist()

        return top_k_indices, top_k_scores


@dataclass(frozen=True)
class RetrievalOutput:
    """Output from a retriever operation.

    Args:
        results: List of retrieval results sorted by relevance.
        latency_ms: Total latency in milliseconds.
    """
    results: list[RetrievalResult]
    latency_ms: float


class DenseRetriever:
    """Dense vector retriever using embeddings and vector similarity search.

    Args:
        embedder: OllamaEmbedder instance for text embedding.
        backend: VectorBackend instance (e.g., NumpyBackend) for similarity search.
    """

    def __init__(
        self, embedder: OllamaEmbedder, backend: VectorBackend
    ) -> None:
        self._embedder = embedder
        self._backend = backend
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks by embedding them and building the backend.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = list(chunks)

        if not chunks:
            # Empty collection - build empty index
            self._backend.build(np.empty((0, self._embedder.dimension)))
            return

        # Embed all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_batch(texts)

        # Convert to numpy array
        matrix = np.array(embeddings)

        # Build backend index
        self._backend.build(matrix)

    def retrieve(self, query: str, k: int) -> RetrievalOutput:
        """Retrieve top-k most relevant chunks for a query.

        Args:
            query: Query text string.
            k: Number of results to return. If k < 1, returns empty results.
                If k > number of indexed chunks, returns all chunks.

        Returns:
            RetrievalOutput with results and latency in milliseconds.

        Raises:
            ValueError: If query embedding has zero magnitude or dimension mismatch.
        """
        start_time = time.perf_counter()

        # Handle empty collection case
        if not self._chunks:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalOutput(results=[], latency_ms=latency_ms)

        # Embed query
        query_embedding = self._embedder.embed(query)
        query_vec = np.array(query_embedding)

        # Search backend
        indices, scores = self._backend.search(query_vec, k)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    doc_id=chunk.chunk_id,
                    score=score,
                    rank=rank,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return RetrievalOutput(results=results, latency_ms=latency_ms)
