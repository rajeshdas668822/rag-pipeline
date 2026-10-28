"""Labelled evaluation dataset loader for the RAG reranking lab.

An evaluation dataset is a JSONL file (one JSON object per line) describing
queries paired with the corpus ids judged relevant to them. It is the ground
truth used to compute retrieval/reranking metrics (recall@k, nDCG, MRR, etc.).

Record schema (one JSON object per non-blank line):
    {
        "query_id": "q001",                 # str, unique across the file
        "query": "What is reranking?",      # str, non-empty
        "relevant_documents": ["Mastering RAG 2026_0"],  # list[str], may be []
        "graded_labels": {"Mastering RAG 2026_0": 3}      # optional dict id->int 0..3
    }

ID granularity
--------------
The ids in ``relevant_documents`` (and the keys of ``graded_labels``) may be
either **document-level** ids (``<filename_stem>_<seq>`` e.g.
``Mastering RAG 2026_0``, ``README_0``) or **chunk-level** ids
(``<doc_id>_<chunk_seq>`` e.g. ``Mastering RAG 2026_0_123``). Chunk ids shift
with chunk size/overlap, so the shipped dataset labels at the document level
for stability. Because of this, when a caller supplies ``corpus_ids`` for
membership validation it should include the **document ids** (and optionally
chunk ids) so document-level labels validate correctly.

If ``corpus_ids`` is ``None`` the corpus-membership check is skipped entirely,
so the loader is usable before any corpus has been indexed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "EvaluationQuery",
    "InvalidEvaluationRecordError",
    "load_evaluation_queries",
]


@dataclass(frozen=True)
class EvaluationQuery:
    """A single labelled evaluation query.

    Args:
        query_id: Unique identifier for the query across the dataset.
        query: The natural-language query text (non-empty).
        relevant_documents: List of corpus doc/chunk ids judged relevant.
            May be empty for "no-answer" queries.
        graded_labels: Optional mapping from id to a graded relevance label
            in the range 0..3 (0 = irrelevant, 3 = highly relevant).
    """

    query_id: str
    query: str
    relevant_documents: list[str]
    graded_labels: Optional[dict[str, int]]


class InvalidEvaluationRecordError(ValueError):
    """Raised when an evaluation record is malformed or inconsistent."""


def load_evaluation_queries(
    path: str,
    corpus_ids: Optional[set[str]] = None,
) -> list[EvaluationQuery]:
    """Load and validate a labelled evaluation dataset from a JSONL file.

    Parses one JSON object per non-blank line and validates each record. See
    the module docstring for the record schema and id-granularity notes.

    Validation (each failure raises ``InvalidEvaluationRecordError`` naming the
    offending record and field):
      * ``query_id`` must be present, a non-empty string, and unique.
      * ``query`` must be present and a non-empty (non-whitespace) string.
      * ``relevant_documents`` must be a list of strings. When ``corpus_ids``
        is provided, every id must be a member of that set. When ``corpus_ids``
        is ``None`` the membership check is skipped.
      * ``graded_labels`` (optional) must be a dict mapping str -> int with
        every value in the inclusive range 0..3.

    Args:
        path: Path to the ``.jsonl`` evaluation dataset.
        corpus_ids: Optional set of known corpus ids (doc-level and/or
            chunk-level) used to validate relevant-document membership. If
            ``None``, membership validation is skipped.

    Returns:
        List of validated ``EvaluationQuery`` records in file order.

    Raises:
        InvalidEvaluationRecordError: If any record is malformed or inconsistent.
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    content = file_path.read_text(encoding="utf-8")

    queries: list[EvaluationQuery] = []
    seen_ids: dict[str, int] = {}

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise InvalidEvaluationRecordError(
                f"Line {line_no}: invalid JSON ({e})"
            ) from e

        if not isinstance(record, dict):
            raise InvalidEvaluationRecordError(
                f"Line {line_no}: record must be a JSON object, got {type(record).__name__}"
            )

        # --- query_id ---
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise InvalidEvaluationRecordError(
                f"Line {line_no}: field 'query_id' must be a non-empty string"
            )
        if query_id in seen_ids:
            raise InvalidEvaluationRecordError(
                f"Line {line_no}: duplicate 'query_id' {query_id!r} "
                f"(first seen on line {seen_ids[query_id]})"
            )

        # --- query ---
        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise InvalidEvaluationRecordError(
                f"query_id {query_id!r} (line {line_no}): field 'query' must be a non-empty string"
            )

        # --- relevant_documents ---
        relevant_documents = record.get("relevant_documents", [])
        if not isinstance(relevant_documents, list) or not all(
            isinstance(d, str) for d in relevant_documents
        ):
            raise InvalidEvaluationRecordError(
                f"query_id {query_id!r} (line {line_no}): field 'relevant_documents' "
                f"must be a list of strings"
            )
        if corpus_ids is not None:
            for doc_id in relevant_documents:
                if doc_id not in corpus_ids:
                    raise InvalidEvaluationRecordError(
                        f"query_id {query_id!r} (line {line_no}): relevant document id "
                        f"{doc_id!r} matches no corpus document"
                    )

        # --- graded_labels (optional) ---
        graded_labels: Optional[dict[str, int]] = record.get("graded_labels")
        if graded_labels is not None:
            if not isinstance(graded_labels, dict):
                raise InvalidEvaluationRecordError(
                    f"query_id {query_id!r} (line {line_no}): field 'graded_labels' "
                    f"must be an object mapping id -> int"
                )
            for label_id, label in graded_labels.items():
                if not isinstance(label_id, str):
                    raise InvalidEvaluationRecordError(
                        f"query_id {query_id!r} (line {line_no}): 'graded_labels' key "
                        f"{label_id!r} must be a string"
                    )
                # bool is a subclass of int; reject it explicitly.
                if isinstance(label, bool) or not isinstance(label, int):
                    raise InvalidEvaluationRecordError(
                        f"query_id {query_id!r} (line {line_no}): 'graded_labels' value "
                        f"for {label_id!r} must be an integer"
                    )
                if label < 0 or label > 3:
                    raise InvalidEvaluationRecordError(
                        f"query_id {query_id!r} (line {line_no}): 'graded_labels' value "
                        f"for {label_id!r} is {label}, must be in range 0..3"
                    )
                if corpus_ids is not None and label_id not in corpus_ids:
                    raise InvalidEvaluationRecordError(
                        f"query_id {query_id!r} (line {line_no}): 'graded_labels' id "
                        f"{label_id!r} matches no corpus document"
                    )

        seen_ids[query_id] = line_no
        queries.append(
            EvaluationQuery(
                query_id=query_id,
                query=query,
                relevant_documents=list(relevant_documents),
                graded_labels=dict(graded_labels) if graded_labels is not None else None,
            )
        )

    return queries
