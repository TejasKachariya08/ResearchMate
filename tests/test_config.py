"""Tests for configuration and environment handling in ResearchMate."""

import pytest
from researchmate.config import (
    Settings,
    ensure_google_api_key,
    get_index_name,
    get_index_prefix,
    sanitize_topic_key,
)


def test_settings_load_defaults(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_INDEX_BASENAME", raising=False)

    s = Settings.from_env()
    assert s.gemini_chat_model == "models/gemini-3.8-flash"
    assert s.gemini_embedding_model == "models/gemini-embedding-2"
    assert s.embedding_dimensions == 3072
    assert s.redis_url == "redis://localhost:6379"
    assert s.redis_index_basename == "researchmate"


def test_settings_custom_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "custom-secret-key")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("REDIS_URL", "redis://custom-host:6380")
    monkeypatch.setenv("REDIS_INDEX_BASENAME", "custom_mate")

    s = Settings.from_env()
    assert s.google_api_key == "custom-secret-key"
    assert s.gemini_chat_model == "models/gemini-3.6-flash"
    assert s.gemini_embedding_model == "models/gemini-embedding-2"
    assert s.embedding_dimensions == 1536
    assert s.redis_url == "redis://custom-host:6380"
    assert s.redis_index_basename == "custom_mate"


def test_ensure_google_api_key_valid():
    s = Settings(google_api_key="valid-key")
    assert ensure_google_api_key(s) == "valid-key"


def test_ensure_google_api_key_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s = Settings(google_api_key="")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
        ensure_google_api_key(s)


def test_sanitize_topic_key():
    assert sanitize_topic_key("Quantum Computing & LLMs!") == "quantum_computing_llms"
    assert sanitize_topic_key("   Reinforcement   Learning   ") == "reinforcement_learning"
    assert sanitize_topic_key("---") == "general"


def test_index_naming_and_prefix():
    s = Settings(redis_index_basename="researchmate")
    idx_name = get_index_name("Reinforcement Learning", s)
    prefix = get_index_prefix("Reinforcement Learning", s)

    assert idx_name == "researchmate_reinforcement_learning"
    assert prefix == "researchmate:doc:reinforcement_learning:"
