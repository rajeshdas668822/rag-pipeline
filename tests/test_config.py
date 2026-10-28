"""Unit tests for Ollama Endpoint resolution in :class:`src.config.Config`.

These tests pin down the endpoint precedence order required by the spec:

    ``OLLAMA_ENDPOINT`` env var -> explicit override value -> default.

The ``OLLAMA_ENDPOINT`` environment variable is set/unset via ``monkeypatch``
so each test is isolated from the ambient environment.

Validates: Requirements 17.1, 17.2, 17.3
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import DEFAULT_OLLAMA_ENDPOINT, Config


def test_default_endpoint_applies_when_env_unset_and_no_override(monkeypatch):
    """Requirement 17.3: default applies when env var unset and no override."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load()

    assert config.ollama_endpoint == DEFAULT_OLLAMA_ENDPOINT
    assert config.ollama_endpoint == "http://localhost:11434"


def test_default_endpoint_applies_with_empty_overrides(monkeypatch):
    """Requirement 17.3: an empty overrides mapping still yields the default."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load({})

    assert config.ollama_endpoint == DEFAULT_OLLAMA_ENDPOINT


def test_override_value_used_when_env_unset(monkeypatch):
    """Requirement 17.2: explicit config value overrides the default."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load({"ollama_endpoint": "http://config-host:9999"})

    assert config.ollama_endpoint == "http://config-host:9999"


def test_env_var_takes_precedence_over_override(monkeypatch):
    """Requirement 17.1: env var overrides the explicit config value."""
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://env-host:1234")

    config = Config.load({"ollama_endpoint": "http://config-host:9999"})

    assert config.ollama_endpoint == "http://env-host:1234"


def test_env_var_takes_precedence_over_default(monkeypatch):
    """Requirement 17.1: env var overrides the default when no override given."""
    monkeypatch.setenv("OLLAMA_ENDPOINT", "http://env-host:1234")

    config = Config.load()

    assert config.ollama_endpoint == "http://env-host:1234"


def test_other_field_override_is_applied(monkeypatch):
    """Non-endpoint overrides (e.g. chunk_size) are applied as-is."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load({"chunk_size": 1024, "chunk_overlap": 128})

    assert config.chunk_size == 1024
    assert config.chunk_overlap == 128
    # Untouched fields keep their documented defaults.
    assert config.ollama_endpoint == DEFAULT_OLLAMA_ENDPOINT
    assert config.embedding_model == "embeddinggemma"


def test_endpoint_and_other_overrides_together(monkeypatch):
    """An explicit endpoint and other field overrides coexist correctly."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load(
        {"ollama_endpoint": "http://config-host:9999", "rrf_k": 42}
    )

    assert config.ollama_endpoint == "http://config-host:9999"
    assert config.rrf_k == 42


def test_unknown_override_key_raises_type_error(monkeypatch):
    """An override key that is not a Config field raises TypeError."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    with pytest.raises(TypeError):
        Config.load({"not_a_real_field": "value"})


def test_overrides_mapping_is_not_mutated(monkeypatch):
    """load() must not mutate the caller's overrides mapping."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    overrides = {"ollama_endpoint": "http://config-host:9999"}
    Config.load(overrides)

    # The 'ollama_endpoint' key must still be present in the caller's dict.
    assert overrides == {"ollama_endpoint": "http://config-host:9999"}


def test_config_instance_is_frozen(monkeypatch):
    """The dataclass is frozen; attribute mutation must raise."""
    monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)

    config = Config.load()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ollama_endpoint = "http://mutated:1"  # type: ignore[misc]
