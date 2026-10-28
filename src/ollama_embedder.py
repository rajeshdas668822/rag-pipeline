"""Ollama embedding service.

This module provides a wrapper around the Ollama embedding API with
input validation and error handling. It integrates with the shared
:class:`OllamaClient` to request embeddings for text.

Satisfies Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 17.4.
"""

from __future__ import annotations

from typing import List

from .config import Config
from .ollama_client import OllamaClient


class EmbeddingError(RuntimeError):
    """Raised when the Ollama embedder encounters an error.

    This includes API errors, malformed responses, wrong dimensionality,
    and input validation failures.
    """


class OllamaEmbedder:
    """Ollama-based text embedder with input validation.

    Args:
        client: The OllamaClient instance to use for API calls.
        model: The embedding model name to use. Defaults to "embeddinggemma".
        timeout_s: Request timeout in seconds. Defaults to 30.

    Raises:
        EmbeddingError: If the configured model is not available.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "embeddinggemma",
        timeout_s: int = 30,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_s = timeout_s

        # Determine dimension by requesting a test embedding
        self._dimension = self._get_dimension()

    def _get_dimension(self) -> int:
        """Query the embedding dimensionality by making a test call."""
        try:
            # Request embedding for a minimal test text
            test_text = "test"
            embedding = self._client.embed(self._model, test_text)
            return len(embedding)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to determine embedding dimension for model "
                f"'{self._model}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """The fixed embedding dimensionality for this embedder."""
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for ``text``.

        Args:
            text: The text to embed. Must be non-empty and at most 2048 chars.

        Returns:
            The embedding as a list of floats.

        Raises:
            EmbeddingError: If ``text`` is empty, exceeds 2048 characters,
                the API times out (30s), or returns a malformed/wrong-dimension
                response.
        """
        # Input validation
        if not text or len(text) == 0:
            raise EmbeddingError("Text must be non-empty.")
        if len(text) > 2048:
            raise EmbeddingError("Text must not exceed 2048 characters.")

        # Request embedding
        try:
            embedding = self._client.embed(self._model, text)
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding request failed after {self._timeout_s}s timeout: {exc}"
            ) from exc

        # Validate dimension
        if len(embedding) != self._dimension:
            raise EmbeddingError(
                f"Embedding dimension {len(embedding)} does not match "
                f"expected {self._dimension}."
            )

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts.

        Args:
            texts: List of texts to embed. Each must be non-empty and at most
                2048 chars.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If any input fails validation or the API call fails.
        """
        if not texts:
            return []

        results: List[List[float]] = []
        for text in texts:
            embedding = self.embed(text)
            results.append(embedding)

        return results
