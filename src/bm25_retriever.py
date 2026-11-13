"""BM25 lexical retriever.

Interchangeable with DenseRetriever (same index/retrieve interface).
Satisfies Requirement 12.2.
"""

from __future__ import annotations

import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from .dense_retriever import RetrievalOutput
from .rag_types import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"\w+")

MIN_K = 1
MAX_K = 1000


def tokenize(text: str) -> list[str]:
    """Lowercase text and split into `\\w+` tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lexical retriever ranking chunks with BM25Okapi."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks. An empty list yields an empty index."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])

    def retrieve(self, query: str, k: int) -> RetrievalOutput:
        """Return up to k chunks ranked by descending BM25 score.

        Ties are broken by ascending index. k > N returns all N.

        Raises:
            ValueError: If k is outside 1..1000.
        """
        if k < MIN_K or k > MAX_K:
            raise ValueError(
                f"k must be between {MIN_K} and {MAX_K} inclusive, got {k}."
            )

        start_time = time.perf_counter()

        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RetrievalOutput(results=[], latency_ms=latency_ms)

        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        indices = np.arange(len(scores))
        order = np.lexsort((indices, -scores))[: min(k, len(scores))]

        results = [
            RetrievalResult(
                doc_id=self._chunks[int(i)].chunk_id,
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        return RetrievalOutput(results=results, latency_ms=latency_ms)
