"""Local answer generation via Ollama chat API with citation support.

Builds a context+query prompt instructing the model to answer only from
the supplied chunks and cite supporting document IDs. Returns a structured
result with answer text and citations. Short-circuits on empty context
or when the answer cannot be determined from the context.

Satisfies Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 17.4.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from .ollama_client import OllamaClient
from .rag_types import Chunk


class GenerationError(RuntimeError):
    """Raised when answer generation fails with no fabricated answer."""


@dataclass(frozen=True)
class GenerationResult:
    """Result from a single answer generation call.

    Args:
        answer: The generated answer text.
        citations: List of doc_ids cited from the supplied chunks.
        insufficient_context: True if the model reported the answer
            cannot be determined from the context.
        latency_ms: Total generation time in milliseconds.
    """
    answer: str
    citations: list[str]
    insufficient_context: bool
    latency_ms: float


GENERATION_PROMPT_TEMPLATE = (
    "You are a helpful assistant answering questions strictly from the provided "
    "context. Use only the information in the context below. Do not use any "
    "external knowledge.\n"
    "\n"
    "Instructions:\n"
    "1. Answer the question directly and concisely.\n"
    "2. If the answer cannot be determined from the context, reply with: "
    "'cannot be determined from the context'.\n"
    "3. For each fact in your answer, include a citation in square brackets "
    "with the document ID, e.g., [doc123].\n"
    "4. Only cite document IDs that appear in the context below.\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {query}\n"
    "\n"
    "Answer:"
)


@dataclass(frozen=True)
class ChunkWithId:
    """Helper to hold chunk text and its ID for prompt building."""
    chunk_id: str
    text: str


def _build_context(chunks: list[Chunk]) -> str:
    """Build the context string from chunks with document IDs."""
    lines = []
    for chunk in chunks:
        lines.append("Document ID: " + chunk.chunk_id)
        lines.append(chunk.text)
        lines.append("")  # blank line between docs
    return "\n".join(lines)


class OllamaGenerator:
    """Local answer generator using an Ollama chat/instruct model.

    Args:
        client: Ollama client connected to the local endpoint.
        model: Instruct model name.
        timeout_s: Per-call timeout in seconds.

    Attributes:
        calls: Number of generation calls made (accumulated).
    """

    def __init__(
        self, client: OllamaClient, model: str = "qwen2.5:3b", timeout_s: int = 60
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_s = timeout_s
        self.calls = 0

    def reset_stats(self) -> None:
        """Reset the calls counter."""
        self.calls = 0

    def generate(self, query: str, chunks: list[Chunk]) -> GenerationResult:
        """Generate an answer for the query using the provided chunks.

        Args:
            query: The user's question.
            chunks: List of Chunk objects to use as context.

        Returns:
            GenerationResult with answer, citations, and timing.

        Raises:
            GenerationError: On timeout or failure with no fabricated answer.
        """
        # Empty chunks -> short-circuit without API call
        if not chunks:
            return GenerationResult(
                answer="no context available",
                citations=[],
                insufficient_context=False,
                latency_ms=0.0,
            )

        start_time = time.perf_counter()

        # Build context string with document IDs
        context = _build_context(chunks)

        # Build prompt
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        self.calls += 1

        try:
            response = self._client.chat(
                self._model,
                prompt,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:
            raise GenerationError(
                "Generation failed after " + str(self.calls) + " call(s): " + str(exc)
            ) from exc

        # Parse response for citations and insufficient context
        answer = response.strip()
        citations = []
        insufficient_context = False

        # Check for "cannot be determined from the context" response
        if "cannot be determined from the context" in answer.lower():
            insufficient_context = True
            # Extract any citations that might appear before this message
            citations = self._extract_citations(answer)

        # Extract citations in format [doc_id] from the answer
        if not insufficient_context:
            citations = self._extract_citations(answer)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerationResult(
            answer=answer,
            citations=citations,
            insufficient_context=insufficient_context,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        """Extract document IDs from citations in format [doc_id]."""
        # Match [doc_id] patterns where doc_id contains alphanumeric, underscore, hyphen
        pattern = r"\[([A-Za-z0-9_-]+)\]"
        matches = re.findall(pattern, text)
        return matches
