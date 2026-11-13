"""LLM reranker that scores each candidate with a local Ollama instruct model.

For every candidate one chat call is made using a single fixed prompt template.
The model rates relevance on a 0-3 scale; the score is parsed from the reply.
Unparseable replies (or failed calls) score 0 with ``parse_failed=True`` and
never abort the run. Ties are broken by the retriever's original order.

Satisfies Requirements 13.1, 13.2, 13.3, 13.4, 17.4.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from .cross_encoder_reranker import RerankerError, RerankOutput
from .ollama_client import OllamaClient
from .rag_types import RerankResult, RetrievalResult

logger = logging.getLogger(__name__)

RERANK_PROMPT_TEMPLATE = (
    "You are a relevance judge for a search system.\n"
    "Rate how well the passage answers the query on a scale from 0 to 3:\n"
    "0 = not relevant\n"
    "1 = slightly relevant\n"
    "2 = mostly relevant\n"
    "3 = fully answers the query\n"
    "\n"
    "Query: {query}\n"
    "\n"
    "Passage: {passage}\n"
    "\n"
    "Reply with a single digit (0, 1, 2 or 3) only, with no other text."
)

_SCORE_PATTERN = re.compile(r"\b([0-3])\b")

_GENERATION_OPTIONS = {"temperature": 0, "seed": 42}


class OllamaLLMReranker:
    """Pointwise 0-3 relevance reranker backed by a local Ollama model.

    Args:
        client: Ollama client used for chat calls.
        model: Instruct model name.
        timeout_s: Per-call timeout in seconds.

    Attributes:
        calls: Model evaluations made, accumulated across rerank() calls.
        parse_failures: Evaluations that were unparseable or failed.
    """

    def __init__(
        self, client: OllamaClient, model: str = "qwen2.5:3b", timeout_s: int = 60
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_s = timeout_s
        self._texts: dict[str, str] = {}
        self.calls = 0
        self.parse_failures = 0

    def reset_stats(self) -> None:
        """Reset the ``calls`` and ``parse_failures`` counters."""
        self.calls = 0
        self.parse_failures = 0

    def index(self, chunks) -> None:
        """Register the chunk text used to score candidates.

        Args:
            chunks: Either a mapping of doc_id -> Chunk or text, or an iterable
                of Chunk objects exposing ``chunk_id`` and ``text``.
        """
        if isinstance(chunks, dict):
            self._texts = {
                str(cid): str(v.text if hasattr(v, "text") else v) for cid, v in chunks.items()
            }
            return
        self._texts = {chunk.chunk_id: chunk.text for chunk in chunks}

    @staticmethod
    def _parse_score(response: str) -> tuple[int, bool]:
        """Return ``(score, parse_failed)`` from a raw model response."""
        match = _SCORE_PATTERN.search((response or "").strip())
        if match is None:
            return 0, True
        return int(match.group(1)), False

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        k: Optional[int] = None,
        texts: Optional[dict[str, str]] = None,
    ) -> RerankOutput:
        """Score each candidate with the LLM and reorder by descending score.

        Raises:
            RerankerError: If ``k`` is not an int or is less than 1.
        """
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise RerankerError(
                f"top-k must be an integer of 1 or greater, got {k!r}"
            )

        if not candidates:
            return RerankOutput(results=[], latency_ms=0.0)

        text_map = texts if texts is not None else self._texts

        start_time = time.perf_counter()
        scored: list[tuple[int, bool]] = []
        for cand in candidates:
            prompt = RERANK_PROMPT_TEMPLATE.format(
                query=query, passage=text_map.get(cand.doc_id, "")
            )
            self.calls += 1
            try:
                response = self._client.chat(
                    self._model,
                    prompt,
                    timeout_s=self._timeout_s,
                    options=dict(_GENERATION_OPTIONS),
                )
                score, failed = self._parse_score(response)
            except Exception as exc:  # noqa: BLE001 - never abort a long run
                logger.warning(
                    "LLM rerank call failed for candidate '%s': %s", cand.doc_id, exc
                )
                score, failed = 0, True
            if failed:
                self.parse_failures += 1
            scored.append((score, failed))

        order = sorted(range(len(candidates)), key=lambda i: (-scored[i][0], i))
        if k is not None:
            order = order[:k]

        results = [
            RerankResult(
                doc_id=candidates[i].doc_id,
                rerank_score=float(scored[i][0]),
                reranked_position=position,
                initial_rank=candidates[i].rank,
                initial_score=candidates[i].score,
                parse_failed=scored[i][1],
            )
            for position, i in enumerate(order, start=1)
        ]
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RerankOutput(results=results, latency_ms=latency_ms)
