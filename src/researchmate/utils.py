"""Diagnostics and connectivity utility functions for ResearchMate."""

from __future__ import annotations

from typing import Tuple, Dict, Any
from redis import Redis
import arxiv
from researchmate.config import get_settings


def check_redis_connection(redis_url: str | None = None) -> Tuple[bool, str]:
    """Test connection to the Redis server."""
    url = redis_url or get_settings().redis_url
    try:
        client = Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3)
        client.ping()
        client.close()
        return True, f"Redis server reachable at {url}"
    except Exception as exc:
        return False, f"Cannot connect to Redis at {url}: {exc}"


def check_gemini_connection(
    api_key: str | None = None,
    model: str | None = None,
) -> Tuple[bool, str]:
    """Test connection and authentication with Google Gemini API."""
    settings = get_settings()
    key = api_key or settings.google_api_key
    model_name = model or settings.gemini_chat_model

    if not key:
        return False, "Google API key is missing. Set GOOGLE_API_KEY in .env."

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=key,
            max_tokens=10,
            temperature=0.0,
        )
        response = llm.invoke("ping")
        return True, f"Gemini API responsive with model {model_name} (sample response: {str(response.content)[:30]}...)"
    except Exception as exc:
        return False, f"Gemini API test failed: {exc}"


def check_arxiv_connection() -> Tuple[bool, str]:
    """Test reachability of arXiv API."""
    try:
        client = arxiv.Client(page_size=1, delay_seconds=1.0, num_retries=1)
        search = arxiv.Search(query="quantum", max_results=1)
        results = list(client.results(search))
        if results:
            return True, f"arXiv API reachable (fetched sample paper: '{results[0].title[:40]}...')"
        return True, "arXiv API reachable (no results returned for sample)"
    except Exception as exc:
        return False, f"arXiv API error: {exc}"


def run_all_diagnostics() -> Dict[str, Any]:
    """Execute complete health and connectivity checks."""
    redis_ok, redis_msg = check_redis_connection()
    gemini_ok, gemini_msg = check_gemini_connection()
    arxiv_ok, arxiv_msg = check_arxiv_connection()

    all_passed = redis_ok and gemini_ok and arxiv_ok

    return {
        "status": "healthy" if all_passed else "unhealthy",
        "redis": {"ok": redis_ok, "message": redis_msg},
        "gemini": {"ok": gemini_ok, "message": gemini_msg},
        "arxiv": {"ok": arxiv_ok, "message": arxiv_msg},
    }
