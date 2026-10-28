"""Shared data types for the RAG reranking lab.

These dataclasses form the contract that makes retrievers and rerankers interchangeable.
All types are frozen (immutable) to ensure data integrity throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Document:
    """A stored unit of knowledge-base text.
    
    Args:
        doc_id: Unique identifier across the entire knowledge base.
        text: Non-empty extracted or recorded text content.
        source_path: File path where this document originated.
        metadata: Optional dictionary of additional metadata.
    """
    doc_id: str
    text: str
    source_path: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A fixed-size, optionally overlapping segment of a document's text.
    
    Args:
        chunk_id: Unique identifier across all chunks in a single run.
        doc_id: Reference to the parent document's id.
        text: The chunk text content.
        seq: Sequence number within the parent document (0-indexed).
    """
    chunk_id: str
    doc_id: str
    text: str
    seq: int


@dataclass(frozen=True)
class RetrievalResult:
    """One entry in a retriever's ranked candidate set.
    
    Args:
        doc_id: The chunk_id or doc_id of the retrieved unit.
        score: The retriever's raw score (cosine similarity, BM25, or RRF).
        rank: 1-based position in the ranked result list.
    """
    doc_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class RerankResult:
    """One entry in a reranker's output.
    
    Args:
        doc_id: The document identifier.
        rerank_score: Raw comparison value from the reranker (not calibrated probability).
        reranked_position: 1-based position after reranking.
        initial_rank: 1-based position in the original candidate set.
        initial_score: The retriever score this candidate entered with.
        parse_failed: For LLM rerankers, True if the response was unparseable.
    """
    doc_id: str
    rerank_score: float
    reranked_position: int
    initial_rank: int
    initial_score: float
    parse_failed: bool = False
