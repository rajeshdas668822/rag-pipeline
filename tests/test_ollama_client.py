"""Unit tests for :meth:`src.ollama_client.OllamaClient.ensure_ready`.

The ``ollama`` package is intentionally **not** installed in this environment,
so these tests never require a live Ollama runtime or the real SDK. Instead a
fake ``ollama`` module exposing a ``Client`` with a ``.list()`` method is
injected via ``monkeypatch``. This exercises the genuine code path through
``_get_client`` -> ``ping`` / ``available_models`` -> ``ensure_ready``, driven
purely by the fake transport's ``.list()`` behaviour (raise to simulate an
unreachable endpoint, return a model listing to simulate installed models).

Validates: Requirements 1.7, 17.5, 17.6
"""

from __future__ import annotations

import sys
import types

import pytest

from src.ollama_client import (
    OllamaClient,
    OllamaModelMissingError,
    OllamaUnavailableError,
)

ENDPOINT = "http://localhost:11434"


class _FakeModelEntry:
    """Mimics a typed Ollama model entry exposing a ``.model`` attribute."""

    def __init__(self, name: str) -> None:
        self.model = name


class _FakeListResponse:
    """Mimics the typed ``list()`` response exposing ``.models``."""

    def __init__(self, names: list[str]) -> None:
        self.models = [_FakeModelEntry(n) for n in names]


class _FakeClient:
    """A stand-in for ``ollama.Client`` whose ``.list()`` is scripted.

    Args:
        model_names: Names the endpoint reports as installed.
        reachable: When ``False``, ``.list()`` raises to simulate an
            unreachable endpoint (mirrors a transport/connection error).
    """

    def __init__(
        self,
        *,
        model_names: list[str],
        reachable: bool = True,
        **_kwargs: object,
    ) -> None:
        self._model_names = model_names
        self._reachable = reachable

    def list(self) -> _FakeListResponse:
        if not self._reachable:
            raise ConnectionError("simulated: endpoint not reachable")
        return _FakeListResponse(self._model_names)


def _install_fake_ollama(
    monkeypatch, *, model_names: list[str], reachable: bool = True
) -> None:
    """Inject a fake ``ollama`` module so the lazy ``from ollama import Client``
    inside ``OllamaClient._get_client`` resolves to :class:`_FakeClient`."""

    def _client_factory(*args: object, **kwargs: object) -> _FakeClient:
        return _FakeClient(model_names=model_names, reachable=reachable, **kwargs)

    fake_module = types.ModuleType("ollama")
    fake_module.Client = _client_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake_module)


def test_ensure_ready_returns_none_when_reachable_and_models_present(monkeypatch):
    """Requirement 17.5/17.6: reachable endpoint with all required models
    present returns ``None`` and raises nothing."""
    _install_fake_ollama(
        monkeypatch, model_names=["embeddinggemma:latest", "qwen2.5:3b"]
    )
    client = OllamaClient(ENDPOINT)

    result = client.ensure_ready(["embeddinggemma", "qwen2.5:3b"])

    assert result is None


def test_ensure_ready_raises_unavailable_naming_endpoint_when_unreachable(
    monkeypatch,
):
    """Requirement 1.7/17.6: an unreachable endpoint raises
    ``OllamaUnavailableError`` and the message names the configured endpoint."""
    _install_fake_ollama(monkeypatch, model_names=[], reachable=False)
    client = OllamaClient(ENDPOINT)

    with pytest.raises(OllamaUnavailableError) as exc_info:
        client.ensure_ready(["embeddinggemma"])

    assert ENDPOINT in str(exc_info.value)


def test_ensure_ready_raises_model_missing_naming_endpoint_and_model(monkeypatch):
    """Requirement 1.7/17.6: a reachable endpoint missing a required model
    raises ``OllamaModelMissingError`` naming both the endpoint and model."""
    _install_fake_ollama(monkeypatch, model_names=["embeddinggemma:latest"])
    client = OllamaClient(ENDPOINT)

    with pytest.raises(OllamaModelMissingError) as exc_info:
        client.ensure_ready(["embeddinggemma", "qwen2.5:3b"])

    message = str(exc_info.value)
    assert ENDPOINT in message
    assert "qwen2.5:3b" in message


def test_ensure_ready_tag_tolerant_match_for_bare_name(monkeypatch):
    """A bare required name 'embeddinggemma' is satisfied by an installed
    'embeddinggemma:latest' (tag-insensitive match)."""
    _install_fake_ollama(monkeypatch, model_names=["embeddinggemma:latest"])
    client = OllamaClient(ENDPOINT)

    assert client.ensure_ready(["embeddinggemma"]) is None


def test_ensure_ready_exact_match_for_pinned_tag(monkeypatch):
    """A pinned required model 'qwen2.5:3b' present exactly is satisfied."""
    _install_fake_ollama(
        monkeypatch, model_names=["qwen2.5:3b", "embeddinggemma:latest"]
    )
    client = OllamaClient(ENDPOINT)

    assert client.ensure_ready(["qwen2.5:3b"]) is None
