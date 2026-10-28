"""Ollama runtime connection and model-availability check using requests.

This module centralises the HTTP connection to the configured Ollama Endpoint
and performs the startup reachability + model-availability check used by every
Ollama-dependent phase. Every component that needs an Ollama model
(:class:`OllamaEmbedder`, :class:`OllamaGenerator`, :class:`OllamaLLMReranker`)
talks to Ollama through this one client, so switching between a native install
and a Docker container is purely a configuration change.

The client connects **only** to the configured local endpoint. It never falls
back to a cloud-hosted or external/paid inference API: if the endpoint is
unreachable or a required model is absent, the relevant error is raised naming
the endpoint (and the missing model) instead.

Uses the `requests` library directly - no dependency on the `ollama` Python package.

Satisfies Requirements 1.6, 1.7, 17.4, 17.5, 17.6, 17.8.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import DEFAULT_OLLAMA_ENDPOINT


class OllamaUnavailableError(RuntimeError):
    """Raised when the configured Ollama Endpoint cannot be reached.

    The message names the configured endpoint. The system never falls
    back to an external or paid API when this is raised. (Requirements 1.7, 17.6)
    """


class OllamaModelMissingError(RuntimeError):
    """Raised when a required model is absent at the Ollama Endpoint.

    The message names both the configured endpoint and the missing model. The
    system never falls back to an external or paid API when this is raised.
    (Requirements 1.7, 17.6)
    """


def _normalize_model_name(name: str) -> str:
    """Return a model name without its tag for tolerant comparison."""
    return name.split(":", 1)[0]


class OllamaClient:
    """HTTP client for a locally-running Ollama instance using requests.

    The client connects only to ``endpoint`` (a local base URL such as
    ``http://localhost:11434``) and performs reachability and model-availability
    checks. It never contacts a cloud-hosted or external Ollama service.
    """

    def __init__(self, endpoint: str, *, timeout_s: int = 30) -> None:
        """Create a client bound to a single local Ollama Endpoint.

        Args:
            endpoint: The configured Ollama Endpoint base URL.
            timeout_s: Default request timeout in seconds.
        """
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = timeout_s

    @property
    def endpoint(self) -> str:
        """The configured Ollama Endpoint this client connects to."""
        return self._endpoint

    def _request(self, method: str, path: str, *, json: dict | None = None) -> dict:
        """Make a request to the Ollama API and return JSON response.

        Raises OllamaUnavailableError on any network/timeout/error.
        """
        url = f"{self._endpoint}{path}"
        try:
            response = requests.request(
                method, url, json=json, timeout=self._timeout_s
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OllamaUnavailableError(
                f"Request to Ollama Endpoint '{self._endpoint}' failed: {exc}"
            ) from exc

    def ping(self) -> bool:
        """Return ``True`` if the Ollama Endpoint responds, ``False`` otherwise.

        Any connection, timeout, or protocol error is treated as "unreachable"
        and reported as ``False``; this method never raises and never falls
        back to a remote API.
        """
        try:
            self._request("GET", "/api/tags")
            return True
        except Exception:
            return False

    def available_models(self) -> list[str]:
        """Return the names of the models installed at the Ollama Endpoint.

        Returns:
            A list of model names as reported by the endpoint (including their
            tags, for example ``embeddinggemma:latest``).

        Raises:
            OllamaUnavailableError: If the endpoint cannot be reached.
        """
        response = self._request("GET", "/api/tags")
        models = response.get("models", [])
        return [m.get("name", "") for m in models if m.get("name")]

    def ensure_ready(self, required_models: list[str]) -> None:
        """Verify the endpoint is reachable and all required models are present.

        Args:
            required_models: Model names that must be installed at the endpoint.

        Raises:
            OllamaUnavailableError: If the endpoint cannot be reached.
            OllamaModelMissingError: If a required model is absent.
        """
        if not self.ping():
            raise OllamaUnavailableError(
                f"The configured Ollama Endpoint '{self._endpoint}' is not "
                "reachable. Start a local Ollama runtime (native install or "
                "Docker container) at this endpoint. No external or paid API "
                "fallback is used."
            )

        installed = self.available_models()
        installed_normalized = {_normalize_model_name(n) for n in installed}

        for required in required_models:
            if _normalize_model_name(required) not in installed_normalized:
                raise OllamaModelMissingError(
                    f"Required model '{required}' is not available at the "
                    f"configured Ollama Endpoint '{self._endpoint}'. Pull it with "
                    f"`ollama pull {required}`. No external or paid API fallback "
                    "is used."
                )

    def embed(self, model: str, text: str) -> list[float]:
        """Return the embedding vector for ``text`` from the endpoint.

        Args:
            model: The embedding model name.
            text: The text to embed.

        Returns:
            The embedding as a list of floats.

        Raises:
            OllamaUnavailableError: If the endpoint cannot be reached.
        """
        response = self._request(
            "POST", "/api/embeddings", json={"model": model, "prompt": text}
        )
        if "embedding" not in response:
            raise OllamaUnavailableError(
                f"The Ollama Endpoint '{self._endpoint}' returned no embedding "
                f"for model '{model}'."
            )
        return [float(x) for x in response["embedding"]]

    def chat(self, model: str, prompt: str, *, timeout_s: int) -> str:
        """Send a single-prompt chat request and return the response text.

        Args:
            model: The chat/instruct model name.
            prompt: The user prompt.
            timeout_s: Per-call request timeout in seconds.

        Returns:
            The assistant message content as a string.

        Raises:
            OllamaUnavailableError: If the endpoint cannot be reached.
        """
        client = OllamaClient(self._endpoint, timeout_s=timeout_s)
        response = client._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        if "message" not in response or "content" not in response["message"]:
            raise OllamaUnavailableError(
                f"The Ollama Endpoint '{self._endpoint}' returned no message "
                f"content for model '{model}'."
            )
        return response["message"]["content"]
