"""Runtime configuration and Ollama Endpoint resolution.

A single :class:`Config` object resolves all runtime settings once, most
importantly the Ollama Endpoint, which is shared by every Ollama-dependent
component (embedder, generator, LLM reranker). The endpoint is resolved in a
fixed precedence order so switching between a native Ollama install and a
Docker container is purely a configuration change.

Endpoint resolution order:
    ``OLLAMA_ENDPOINT`` env var -> explicit config value -> ``DEFAULT_OLLAMA_ENDPOINT``.

Satisfies Requirements 17.1, 17.2, 17.3, 17.8 (plus chunk/model defaults
consumed by later phases).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Optional

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration for the lab.

    The Ollama Endpoint is resolved by :meth:`load`; all other fields carry
    the documented defaults unless overridden.
    """

    ollama_endpoint: str  # env OLLAMA_ENDPOINT -> override value -> default
    embedding_model: str = "embeddinggemma"
    instruct_model: str = "qwen2.5:3b"
    chat_model: str = "qwen2.5:3b"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    dense_backend: str = "numpy"  # "numpy" | "faiss"
    use_ragas: bool = False
    rrf_k: int = 60
    embed_timeout_s: int = 30
    generate_timeout_s: int = 60

    @classmethod
    def load(cls, overrides: Optional[dict] = None) -> "Config":
        """Build a :class:`Config`, resolving the Ollama Endpoint by precedence.

        The Ollama Endpoint is resolved in order:
        ``OLLAMA_ENDPOINT`` env var -> ``overrides["ollama_endpoint"]`` ->
        :data:`DEFAULT_OLLAMA_ENDPOINT`. Any other keys in ``overrides`` that
        name a :class:`Config` field are applied as-is.

        Args:
            overrides: Optional mapping of field names to values. An
                ``ollama_endpoint`` entry acts as the explicit config value in
                the precedence order (and is ignored when the env var is set).

        Returns:
            A frozen :class:`Config` instance.

        Raises:
            TypeError: If ``overrides`` contains a key that is not a
                :class:`Config` field.
        """
        overrides = dict(overrides) if overrides else {}

        valid_fields = {f.name for f in fields(cls)}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise TypeError(
                "Unknown Config override(s): " + ", ".join(sorted(unknown))
            )

        # Endpoint precedence: env var -> override value -> default.
        explicit_endpoint = overrides.pop("ollama_endpoint", None)
        ollama_endpoint = (
            os.environ.get("OLLAMA_ENDPOINT")
            or explicit_endpoint
            or DEFAULT_OLLAMA_ENDPOINT
        )

        return cls(ollama_endpoint=ollama_endpoint, **overrides)
