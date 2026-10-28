"""Cross-encoder reranker executed locally via Sentence Transformers.

This module implements the Phase 2 cross-encoder reranker. It loads
``cross-encoder/ms-marco-MiniLM-L6-v2`` locally through Sentence Transformers
(no Ollama, no paid external API) and re-scores a retriever's candidate set by
pairing the query with each candidate's actual chunk text.

Satisfies Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

from .rag_types import RerankResult, RetrievalResult

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - import guard
    CrossEncoder = None


class RerankerError(RuntimeError):
    """Base error for reranking failures (e.g. invalid top-k)."""


class ModelLoadError(RerankerError):
    """Raised when the cross-encoder model cannot be loaded."""


@dataclass(frozen=True)
class RerankOutput:
    """Output from a reranker operation.

    Args:
        results: Reranked results sorted by descending relevance score.
        latency_ms: Total reranking latency in milliseconds.
    """

    results: list[RerankResult]
    latency_ms: float


class Reranker(Protocol):
    """Protocol for rerankers so implementations are interchangeable."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        k: Optional[int] = None,
    ) -> RerankOutput:
        """Re-score and reorder a candidate set for a query."""
        ...


class CrossEncoderReranker:
    """Cross-encoder reranker using Sentence Transformers.

    The reranker scores the query against each candidate's chunk *text*, not its
    id. Because a :class:`RetrievalResult` only carries the chunk's ``doc_id``,
    the caller must supply the id->text mapping. Provide it either by calling
    :meth:`index` once with the chunk map, or by passing ``texts`` directly to
    :meth:`rerank`. The per-call ``texts`` argument takes precedence.

    Args:
        model_name: Hugging Face model id to load locally.

    Raises:
        ModelLoadError: If Sentence Transformers is unavailable or the model
            fails to load. No results are produced in that case.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ) -> None:
        if CrossEncoder is None:
            raise ModelLoadError(
                "sentence-transformers is not installed; cannot load the "
                f"cross-encoder model '{model_name}'."
            )
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 - surface any load failure
            raise ModelLoadError(
                f"Failed to load cross-encoder model '{model_name}': {exc}"
            ) from exc
        self._texts: dict[str, str] = {}

    def index(self, chunks) -> None:
        """Register the chunk text used to score candidates.

        Args:
            chunks: Either a mapping of doc_id -> text, or an iterable of Chunk
                objects exposing ``chunk_id`` and ``text`` attributes.
        """
        if isinstance(chunks, dict):
            self._texts = {str(cid): str(text) for cid, text in chunks.items()}
            return
        self._texts = {chunk.chunk_id: chunk.text for chunk in chunks}

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        k: Optional[int] = None,
        texts: Optional[dict[str, str]] = None,
    ) -> RerankOutput:
        """Re-score and reorder a candidate set with the cross-encoder.

        Builds exactly one (query, chunk_text) pair per candidate and computes
        one raw relevance score for each. Results are sorted by descending score
        with a stable tie order (original candidate order is preserved on ties).

        Args:
            query: The user query.
            candidates: Candidate set from a retriever. Each ``doc_id`` is the
                chunk id whose text is looked up via ``texts`` / :meth:`index`.
            k: Optional top-k. When given, the first ``min(k, len)`` results are
                returned. Must be an int >= 1.
            texts: Optional per-call doc_id -> text mapping. Overrides the map
                set via :meth:`index` when provided.

        Returns:
            RerankOutput with results and latency in milliseconds. Scores are
            raw comparison values, not calibrated probabilities.

        Raises:
            RerankerError: If ``k`` is not an int or is less than 1.
        """
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise RerankerError(
                f"top-k must be an integer of 1 or greater, got {k!r}"
            )

        if not candidates:
            return RerankOutput(results=[], latency_ms=0.0)

        text_map = texts if texts is not None else self._texts

        # Exactly one (query, chunk_text) pair per candidate. Missing text falls
        # back to an empty string so the candidate still gets a score.
        pairs = [(query, text_map.get(c.doc_id, "")) for c in candidates]

        start_time = time.perf_counter()
        scores = self._model.predict(pairs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Sort by descending score; stable tie order via original index.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(scores[i]), i),
        )

        if k is not None:
            order = order[:k]

        results = [
            RerankResult(
                doc_id=candidates[i].doc_id,
                rerank_score=float(scores[i]),
                reranked_position=position,
                initial_rank=candidates[i].rank,
                initial_score=candidates[i].score,
                parse_failed=False,
            )
            for position, i in enumerate(order, start=1)
        ]

        return RerankOutput(results=results, latency_ms=latency_ms)
