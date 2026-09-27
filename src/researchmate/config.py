"""Configuration module for ResearchMate.

Handles environment variables, API keys, model selections,
and Redis index naming conventions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""

    google_api_key: str = ""
    gemini_chat_model: str = "models/gemini-3.8-flash"
    gemini_embedding_model: str = "models/gemini-embedding-2"
    embedding_dimensions: int = 3072
    redis_url: str = "redis://localhost:6379"
    redis_index_basename: str = "researchmate"
    tokenizers_parallelism: bool = False
    debug: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables with sensible defaults."""
        raw_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        chat_model = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-3.8-flash").strip()
        if not chat_model.startswith("models/"):
            chat_model = f"models/{chat_model}"

        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2").strip()
        if not embedding_model.startswith("models/"):
            embedding_model = f"models/{embedding_model}"

        try:
            emb_dims = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        except ValueError:
            emb_dims = 3072

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379").strip()
        basename = os.getenv("REDIS_INDEX_BASENAME", "researchmate").strip()
        parallelism = os.getenv("TOKENIZERS_PARALLELISM", "false").lower() in ("true", "1")
        debug = os.getenv("RESEARCHMATE_DEBUG", "false").lower() in ("true", "1")

        return cls(
            google_api_key=raw_api_key,
            gemini_chat_model=chat_model,
            gemini_embedding_model=embedding_model,
            embedding_dimensions=emb_dims,
            redis_url=redis_url,
            redis_index_basename=basename,
            tokenizers_parallelism=parallelism,
            debug=debug,
        )


_cached_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Get the active application settings singleton."""
    global _cached_settings
    if _cached_settings is None or reload:
        _cached_settings = Settings.from_env()
    return _cached_settings


def ensure_google_api_key(settings: Settings | None = None) -> str:
    """Validate and return the Google API key."""
    current_settings = settings or get_settings()
    api_key = current_settings.google_api_key or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please configure your Google AI Studio API key in the .env file."
        )
    return api_key


def sanitize_topic_key(topic: str) -> str:
    """Convert arbitrary topic string to a safe identifier for Redis keys/indices."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", topic.strip().lower())
    return normalized.strip("_") or "general"


def get_index_name(topic: str, settings: Settings | None = None) -> str:
    """Generate a clean, topic-scoped Redis index name."""
    s = settings or get_settings()
    prefix = s.redis_index_basename or "researchmate"
    clean_topic = sanitize_topic_key(topic)
    return f"{prefix}_{clean_topic}"


def get_index_prefix(topic: str, settings: Settings | None = None) -> str:
    """Generate a clean key prefix for Redis documents stored under a topic."""
    s = settings or get_settings()
    prefix = s.redis_index_basename or "researchmate"
    clean_topic = sanitize_topic_key(topic)
    return f"{prefix}:doc:{clean_topic}:"
