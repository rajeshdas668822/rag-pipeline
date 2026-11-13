"""Hybrid retriever fusing dense and BM25 rankings with Reciprocal Rank Fusion.

Interchangeable with DenseRetriever and BM25Retriever (same index/retrieve
interface). Satisfies Requirement 12.3.
"""

from __future__ import annotations

import time

from .dense_retriever import RetrievalOutput
from .rag_types import Chunk, RetrievalResult

MIN_K = 1
MAX_K = 1000


def rrf_fuse(
    ranked_lists: list[list[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with Reciprocal Rank Fusion.

    score(d) = sum over lists containing d of 1 / (rrf_k + rank_d), with
    rank 1-based. Only ranks are used, never raw retriever scores.

    Returns:
        (id, fused_score) pairs sorted by descending score; ties broken by
        best (lowest) rank in any list, then by id.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank[item[0]], item[0]),
    )


class HybridRetriever:
    """Combines a dense and a BM25 retriever using RRF.

    Args:
        dense: Retriever exposing index(chunks) and retrieve(query, k).
        bm25: Retriever exposing index(chunks) and retrieve(query, k).
        rrf_k: RRF smoothing constant.
        fetch_k: Minimum depth fetched from each retriever before fusion.
    """

    def __init__(self, dense, bm25, rrf_k: int = 60, fetch_k: int = 100) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._fetch_k = fetch_k

    def index(self, chunks: list[Chunk]) -> None:
        """Index both underlying retrievers.

        Callers may instead pass already-indexed retrievers to the constructor
        and skip this call (Phase 4 does this so the dense corpus is embedded
        only once).
        """
        self._dense.index(chunks)
        self._bm25.index(chunks)

    def retrieve(self, query: str, k: int) -> RetrievalOutput:
        """Return the top-k chunks by fused RRF score.

        Raises:
            ValueError: If k is outside 1..1000.
        """
        if k < MIN_K or k > MAX_K:
            raise ValueError(
                f"k must be between {MIN_K} and {MAX_K} inclusive, got {k}."
            )

        start_time = time.perf_counter()

        n = min(max(k, self._fetch_k), MAX_K)
        dense_ids = [r.doc_id for r in self._dense.retrieve(query, n).results]
        bm25_ids = [r.doc_id for r in self._bm25.retrieve(query, n).results]

        fused = rrf_fuse([dense_ids, bm25_ids], self._rrf_k)[:k]
        results = [
            RetrievalResult(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(fused, start=1)
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return RetrievalOutput(results=results, latency_ms=latency_ms)
