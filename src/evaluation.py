"""Scientific evaluation metrics for retrieval and reranking.

This module provides the standard information-retrieval metrics used to compare
a Baseline (candidate set unchanged) against a Reranked configuration over an
identical set of queries, documents, and relevance labels: Recall@K,
Precision@K, MRR@K, and NDCG@K. It also carries retrieval and reranking latency
as two distinct fields so the two stages are measured separately.

The metrics operate on plain ``doc_id`` strings (matching
``RetrievalResult.doc_id`` / ``RerankResult.doc_id``) so this module depends on
no model, Ollama, or network access — pure Python and NumPy only.

Conventions and invariants (the correctness properties this math honours):
    * All metrics stay within ``[0, 1]``.
    * Recall@K is non-decreasing in K.
    * An ideal ordering yields NDCG@K == 1.0.
    * Queries with no relevant docs yield Recall == 0.0 and MRR == 0.0; callers
      exclude such queries from averages consistently.
    * An empty ``ranked`` list or a K larger than the list length is computed
      over the available results without raising.

Satisfies Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """A single query's (or an aggregate's) evaluation scores.

    Retrieval and reranking latency are stored as two distinct fields so the
    two pipeline stages are reported separately (Requirement 10.3). For a
    Baseline configuration with no reranking stage, ``reranking_latency_ms`` is
    passed as ``0.0``.

    Args:
        recall_at_5: Recall@5 in ``[0, 1]``.
        recall_at_50: Recall@50 in ``[0, 1]``.
        precision_at_5: Precision@5 in ``[0, 1]``.
        mrr_at_10: Mean reciprocal rank over the top 10 in ``[0, 1]``.
        ndcg_at_5: NDCG@5 in ``[0, 1]`` (graded labels when available, else binary).
        retrieval_latency_ms: Retrieval stage latency in milliseconds.
        reranking_latency_ms: Reranking stage latency in milliseconds (0.0 when absent).
    """

    recall_at_5: float
    recall_at_50: float
    precision_at_5: float
    mrr_at_10: float
    ndcg_at_5: float
    retrieval_latency_ms: float
    reranking_latency_ms: float


class Evaluation:
    """Stateless computation of standard retrieval/reranking metrics.

    All methods take ``ranked`` as a list of ``doc_id`` strings in ranked order
    (position 0 is the top result) and compute over the available results, so a
    K larger than the list length or an empty list never raises.
    """

    @staticmethod
    def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        """Fraction of all relevant docs that appear in the top-K.

        Recall@K = (number of relevant docs in the top-K) / (total relevant docs).

        Returns ``0.0`` when there are no relevant docs (Requirements 10.4, 10.5).
        Computed over ``min(k, len(ranked))`` results, so K beyond the list
        length does not raise and is non-decreasing in K (Requirement 10.6).

        Args:
            ranked: Doc ids in ranked order (position 0 is the top result).
            relevant: Set of relevant doc ids.
            k: Cutoff rank.

        Returns:
            Recall@K in ``[0, 1]``.
        """
        if not relevant:
            return 0.0
        if k < 1 or not ranked:
            return 0.0

        top_k = ranked[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        """Fraction of the top-K results that are relevant.

        Precision@K = (number of relevant docs in the top-K) / K.

        Denominator choice: standard Precision@K divides by ``K``. When the
        ranked list is shorter than ``K``, we divide by the number of available
        results — ``min(k, len(ranked))`` — rather than ``K``, so a short-but-
        perfect list is not penalised for the missing slots and the value stays
        within ``[0, 1]``. This still computes without error when K exceeds the
        list length (Requirement 10.6).

        Args:
            ranked: Doc ids in ranked order.
            relevant: Set of relevant doc ids.
            k: Cutoff rank.

        Returns:
            Precision@K in ``[0, 1]``.
        """
        if k < 1 or not ranked:
            return 0.0

        top_k = ranked[:k]
        denom = len(top_k)  # == min(k, len(ranked))
        if denom == 0:
            return 0.0

        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / denom

    @staticmethod
    def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
        """Reciprocal rank of the first relevant doc within the top-K.

        MRR@K = 1 / (1-based rank of the first relevant doc in the top-K), or
        ``0.0`` when no relevant doc appears in the top-K. Returns ``0.0`` for
        queries with no relevant docs (Requirements 10.4, 10.5). Computed over
        the available results, so K beyond the list length does not raise
        (Requirement 10.6).

        Args:
            ranked: Doc ids in ranked order.
            relevant: Set of relevant doc ids.
            k: Cutoff rank.

        Returns:
            MRR@K in ``[0, 1]``.
        """
        if not relevant:
            return 0.0
        if k < 1 or not ranked:
            return 0.0

        top_k = ranked[:k]
        for position, doc_id in enumerate(top_k, start=1):
            if doc_id in relevant:
                return 1.0 / position
        return 0.0

    @staticmethod
    def ndcg_at_k(ranked: list[str], graded: dict[str, int], k: int) -> float:
        """Normalised discounted cumulative gain over the top-K.

        Gains come from the ``graded`` labels (integers, typically 0..3). DCG is
        ``sum(gain_i / log2(i + 1))`` over 1-based positions ``i``. IDCG is the
        DCG of the ideal ordering of the available gains (the gains present in
        ``graded``, highest first). NDCG = DCG / IDCG, or ``0.0`` when IDCG == 0
        (e.g. no graded docs or all-zero gains). An ideal ordering yields exactly
        ``1.0``.

        Callers wanting binary relevance pass ``{doc_id: 1}`` for each relevant
        doc; :meth:`evaluate` does this automatically when no graded labels are
        supplied.

        Computed over ``min(k, len(ranked))`` positions, so K beyond the list
        length does not raise (Requirement 10.6).

        Args:
            ranked: Doc ids in ranked order.
            graded: Mapping of doc id to integer graded gain.
            k: Cutoff rank.

        Returns:
            NDCG@K in ``[0, 1]``.
        """
        if k < 1 or not graded:
            return 0.0

        top_k = ranked[:k]

        # DCG of the actual ranking: gain / log2(position + 1), 1-based positions.
        dcg = 0.0
        for position, doc_id in enumerate(top_k, start=1):
            gain = graded.get(doc_id, 0)
            if gain:
                dcg += gain / np.log2(position + 1)

        # IDCG: ideal ordering of the available gains, highest gains first,
        # over the same cutoff.
        ideal_gains = sorted(graded.values(), reverse=True)[:k]
        idcg = 0.0
        for position, gain in enumerate(ideal_gains, start=1):
            if gain:
                idcg += gain / np.log2(position + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @classmethod
    def evaluate(
        cls,
        ranked: list[str],
        relevant: set[str],
        graded: Optional[dict[str, int]],
        retrieval_latency_ms: float,
        reranking_latency_ms: float,
    ) -> MetricResult:
        """Compute the full metric suite for one query.

        Populates Recall@5, Recall@50, Precision@5, MRR@10, and NDCG@5. NDCG
        uses the graded labels when ``graded`` is provided and non-empty;
        otherwise it falls back to binary relevance (gain 1 for each relevant
        doc, 0 otherwise). The two latency arguments are stored verbatim as two
        distinct fields (Requirement 10.3); a Baseline config passes
        ``reranking_latency_ms=0.0``.

        Args:
            ranked: Doc ids in ranked order (position 0 is the top result).
            relevant: Set of relevant doc ids.
            graded: Optional mapping of doc id to integer graded gain (0..3).
                When ``None`` or empty, NDCG uses binary relevance.
            retrieval_latency_ms: Retrieval stage latency in milliseconds.
            reranking_latency_ms: Reranking stage latency in milliseconds.

        Returns:
            A :class:`MetricResult` for this query.
        """
        # Choose graded gains when provided and non-empty; else binary relevance.
        if graded:
            ndcg_gains = graded
        else:
            ndcg_gains = {doc_id: 1 for doc_id in relevant}

        return MetricResult(
            recall_at_5=cls.recall_at_k(ranked, relevant, 5),
            recall_at_50=cls.recall_at_k(ranked, relevant, 50),
            precision_at_5=cls.precision_at_k(ranked, relevant, 5),
            mrr_at_10=cls.mrr_at_k(ranked, relevant, 10),
            ndcg_at_5=cls.ndcg_at_k(ranked, ndcg_gains, 5),
            retrieval_latency_ms=retrieval_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
        )
