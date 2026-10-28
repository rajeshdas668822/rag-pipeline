"""Text chunking for RAG retrieval.

Produces fixed-size, optionally overlapping chunks from documents with unique chunk IDs.
"""

from __future__ import annotations

from typing import List

from .rag_types import Chunk, Document


class InvalidChunkConfigError(ValueError):
    """Raised when chunk configuration is invalid.

    Args:
        message: Descriptive error message stating the allowed range.
    """

    pass


class Chunker:
    """Split text into fixed-size, overlapping chunks.

    Args:
        chunk_size: Maximum number of units per chunk (must be >= 1).
        chunk_overlap: Number of units to overlap between consecutive chunks (must be 0 <= overlap < size).

    Raises:
        InvalidChunkConfigError: If chunk_size < 1, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize the chunker with size and overlap parameters."""
        if chunk_size < 1:
            raise InvalidChunkConfigError(f"chunk_size must be >= 1, got {chunk_size}")
        if chunk_overlap < 0:
            raise InvalidChunkConfigError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigError(
                f"chunk_overlap must be < chunk_size, got overlap={chunk_overlap} >= size={chunk_size}"
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> List[Chunk]:
        """Split a document into sequential chunks.

        Args:
            doc: The document to chunk.

        Returns:
            List of Chunk objects. Consecutive chunks share exactly chunk_overlap
            trailing/leading units. Returns empty list if doc.text is empty.
        """
        if not doc.text:
            return []

        text = doc.text
        chunks: List[Chunk] = []
        step = self._chunk_size - self._chunk_overlap

        # Handle case where text fits in one chunk
        if len(text) <= self._chunk_size:
            chunk_id = self._assign_id(doc.doc_id, 0)
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, seq=0))
            return chunks

        # Generate sequential chunks with overlap
        seq = 0
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]
            chunk_id = self._assign_id(doc.doc_id, seq)
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=chunk_text, seq=seq))
            start += step
            seq += 1

        return chunks

    def _assign_id(self, doc_id: str, seq: int) -> str:
        """Generate a unique chunk ID.

        Args:
            doc_id: The parent document ID.
            seq: The sequence number within the document (0-indexed).

        Returns:
            Formatted chunk ID: "<doc_id>_<seq>".
        """
        return f"{doc_id}_{seq}"
