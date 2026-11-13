"""Persistent on-disk embedding cache.

:class:`CachedEmbedder` wraps an :class:`OllamaEmbedder` and stores vectors in
a SQLite database keyed by (model, sha256(text)). Lookups depend only on the
model and the exact text, so changing chunk settings re-embeds only new or
changed chunks, and changing the model never reuses another model's vectors.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .ollama_embedder import EmbeddingError, OllamaEmbedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / ".cache" / "embeddings.sqlite"

# Stay well under SQLite's default variable limit (999 on older builds).
_SQL_CHUNK = 500


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Drop-in caching wrapper exposing ``dimension``, ``embed``, ``embed_batch``.

    Args:
        embedder: The wrapped :class:`OllamaEmbedder`.
        cache_path: SQLite file path. Defaults to ``<project>/.cache/embeddings.sqlite``.
        model: Model name used in the cache key. Defaults to the wrapped
            embedder's model.
    """

    def __init__(
        self,
        embedder: OllamaEmbedder,
        cache_path: Optional[Union[str, Path]] = None,
        model: Optional[str] = None,
    ) -> None:
        self._embedder = embedder
        self._model = model if model is not None else getattr(embedder, "_model")
        self._path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "model TEXT NOT NULL, text_hash TEXT NOT NULL, "
            "dim INTEGER NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (model, text_hash))"
        )
        self._conn.commit()

    @property
    def dimension(self) -> int:
        """Embedding dimensionality of the wrapped embedder."""
        return self._embedder.dimension

    def stats(self) -> Dict[str, int]:
        """Return ``{"hits": N, "misses": M}`` counters."""
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    @staticmethod
    def _validate(text: str) -> None:
        # Same rules as OllamaEmbedder.embed.
        if not text or len(text) == 0:
            raise EmbeddingError("Text must be non-empty.")
        if len(text) > 2048:
            raise EmbeddingError("Text must not exceed 2048 characters.")

    def _lookup(self, hashes: List[str]) -> Dict[str, List[float]]:
        """Return {hash: vector} for valid cached entries (dim must match)."""
        found: Dict[str, List[float]] = {}
        dim = self.dimension
        for i in range(0, len(hashes), _SQL_CHUNK):
            part = hashes[i:i + _SQL_CHUNK]
            marks = ",".join("?" * len(part))
            rows = self._conn.execute(
                f"SELECT text_hash, dim, vector FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({marks})",
                [self._model, *part],
            )
            for h, stored_dim, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                if stored_dim != dim or vec.shape[0] != dim:
                    continue  # stale entry: treat as a miss, overwritten on insert
                found[h] = vec.astype(np.float64).tolist()
        return found

    def _store(self, items: Dict[str, List[float]]) -> None:
        if not items:
            return
        dim = self.dimension
        rows = [
            (self._model, h, dim, np.asarray(v, dtype=np.float32).tobytes())
            for h, v in items.items()
        ]
        with self._conn:  # single transaction
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def embed(self, text: str) -> List[float]:
        """Return the embedding for ``text``, using the cache when possible."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for ``texts`` in input order, computing only misses."""
        if not texts:
            return []
        for text in texts:
            self._validate(text)

        hashes = [_text_hash(t) for t in texts]
        cached = self._lookup(list(dict.fromkeys(hashes)))

        computed: Dict[str, List[float]] = {}
        try:
            for text, h in zip(texts, hashes):
                if h in cached:
                    self.hits += 1
                elif h in computed:
                    self.hits += 1  # duplicate within this batch, computed once
                else:
                    computed[h] = self._embedder.embed(text)
                    self.misses += 1
        finally:
            self._store(computed)  # keep partial progress if a later text fails

        return [list(cached[h]) if h in cached else list(computed[h]) for h in hashes]
