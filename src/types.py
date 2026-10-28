"""Shared data types for the RAG reranking lab.

These dataclasses form the contract that makes retrievers and rerankers interchangeable.
All types are frozen (immutable) to ensure data integrity throughout the pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Document:
    """A stored unit of knowledge-base text.

    Args:
        id: Unique identifier.
        source: Path/URL where the document came from.
        content: Full text content.
        metadata: Extra info (source type, timestamp, etc.).
    """
    id: str
    source: str
    content: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    """A fixed-size, optionally overlapping segment of a document's text.

    Args:
        id: Unique identifier (usually <doc_id>#<seq>).
        document_id: Parent document ID.
        content: Chunk text.
        start_char: Character offset in original document.
        end_char: Character offset in original document.
    """
    id: str
    document_id: str
    content: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class RetrievalResult:
    """One entry in a retriever's ranked candidate set.

    Args:
        document_id: The document identifier.
        score: Similarity/relevance score.
        rank: 1-based position in the ranked result list.
    """
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class RerankResult:
    """One entry in a reranker's output.

    Args:
        document_id: The document identifier.
        score: Reranker's score.
        rank: New position after reranking.
        parse_failed: For LLM-based rerankers, True if parsing failed.
    """
    document_id: str
    score: float
    rank: int
    parse_failed: bool = False
