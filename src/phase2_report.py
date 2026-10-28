"""Phase 2 CSV report writer.

Writes one row per candidate per query describing how the cross-encoder reranker
moved each document. The ``is_relevant`` column comes exclusively from manually
assigned relevance labels and is NEVER derived from reranker scores.

Satisfies Requirements 8.3, 8.4, 8.6.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from .cross_encoder_reranker import RerankOutput
from .rag_types import RetrievalResult

PHASE2_FIELDS = [
    "query_id",
    "document_id",
    "initial_rank",
    "initial_similarity_score",
    "rerank_score",
    "reranked_position",
    "is_relevant",
]


def write_phase2_report(
    output_path: str | Path,
    query_id: str,
    candidates: Iterable[RetrievalResult],
    rerank_output: RerankOutput,
    relevant_doc_ids: set[str],
) -> str:
    """Write the Phase 2 per-candidate-per-query CSV report.

    One row is emitted for every candidate in the initial candidate set, so a
    candidate that was dropped from the top-k reranked output still appears.

    reranked_position policy: if the candidate is present in ``rerank_output``
    (i.e. reranking was computed for it) its actual reranked position is written.
    If a candidate was in the initial set but is absent from the reranked output
    (for example when the reranker was asked for a smaller top-k than the
    candidate count), its ``reranked_position`` cell is left blank.

    Args:
        output_path: Destination CSV path (parent dirs are created).
        query_id: Identifier for the query these candidates belong to.
        candidates: The initial candidate set (RetrievalResult list).
        rerank_output: Output of the cross-encoder reranker for this query.
        relevant_doc_ids: Manually labelled relevant doc ids. Used verbatim for
            the ``is_relevant`` column; never derived from scores.

    Returns:
        The written file path as a string.
    """
    candidates = list(candidates)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Map doc_id -> reranked RerankResult for quick lookup.
    reranked_by_id = {r.doc_id: r for r in rerank_output.results}

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PHASE2_FIELDS)
        writer.writeheader()
        for cand in candidates:
            reranked = reranked_by_id.get(cand.doc_id)
            rerank_score: Optional[float]
            reranked_position: Optional[int]
            if reranked is not None:
                rerank_score = reranked.rerank_score
                reranked_position = reranked.reranked_position
            else:
                rerank_score = None
                reranked_position = None

            writer.writerow(
                {
                    "query_id": query_id,
                    "document_id": cand.doc_id,
                    "initial_rank": cand.rank,
                    "initial_similarity_score": cand.score,
                    "rerank_score": "" if rerank_score is None else rerank_score,
                    "reranked_position": (
                        "" if reranked_position is None else reranked_position
                    ),
                    "is_relevant": cand.doc_id in relevant_doc_ids,
                }
            )

    return str(out)
