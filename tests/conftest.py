"""Pytest shared fixtures and mock environments for ResearchMate."""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_raw_documents():
    """Return raw mock documents as returned by arxiv API."""
    return [
        Document(
            page_content="Reinforcement learning from human feedback aligns language models with human intent.",
            metadata={
                "Title": "Training language models to follow instructions",
                "Authors": "Long Ouyang, Jeff Wu, Xu Jiang",
                "primary_category": "cs.CL",
                "links": "https://arxiv.org/abs/2203.02155",
            },
        ),
        Document(
            page_content="Direct Preference Optimization optimizes language models directly on preferences.",
            metadata={
                "title": "Direct Preference Optimization",
                "authors": "Rafael Rafailov, Archit Sharma",
                "category": "cs.LG",
                "links": "https://arxiv.org/abs/2305.18290",
            },
        ),
    ]


@pytest.fixture
def mock_env(monkeypatch):
    """Set up controlled environment variables."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-mock-key-12345")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "models/gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("REDIS_INDEX_BASENAME", "researchmate")
