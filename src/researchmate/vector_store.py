"""Redis Vector Store management module for ResearchMate.

Provides indexing, schema configuration, document normalization,
and index cleanup for Redis Vector Search.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError, ResponseError

from researchmate.config import (
    Settings,
    get_index_name,
    get_index_prefix,
    get_settings,
)

logger = logging.getLogger(__name__)

METADATA_SCHEMA = [
    {"name": "title", "type": "text"},
    {"name": "authors", "type": "tag"},
    {"name": "category", "type": "tag"},
    {"name": "links", "type": "tag"},
]


def get_redis_client(redis_url: str | None = None) -> Redis:
    """Create a Redis client instance with connection timeouts.

    Args:
        redis_url: Optional Redis connection URL. Defaults to configuration.

    Returns:
        Configured Redis client.
    """
    url = redis_url or get_settings().redis_url
    return Redis.from_url(
        url,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def clean_documents(documents: List[Document]) -> List[Document]:
    """Standardize and clean document metadata for Redis storage.

    Args:
        documents: Raw documents from arXiv fetcher.

    Returns:
        Normalized list of Document objects.
    """
    cleaned: List[Document] = []
    for doc in documents:
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("Title") or "Untitled Document"
        authors = meta.get("authors") or meta.get("Authors") or "Unknown"
        category = meta.get("category") or meta.get("primary_category") or ""
        links = meta.get("links") or meta.get("entry_id") or ""

        cleaned.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "title": str(title),
                    "authors": str(authors),
                    "category": str(category),
                    "links": str(links),
                },
            )
        )
    return cleaned


def get_redis_config(topic: str, settings: Settings | None = None) -> RedisConfig:
    """Construct RedisConfig for topic-scoped vector indexing.

    Args:
        topic: Topic string used to name the index and prefix.
        settings: Optional Settings instance.

    Returns:
        Configured RedisConfig.
    """
    s = settings or get_settings()
    return RedisConfig(
        index_name=get_index_name(topic, s),
        key_prefix=get_index_prefix(topic, s),
        redis_url=s.redis_url,
        storage_type="hash",
        content_field="text",
        embedding_field="embedding",
        metadata_schema=METADATA_SCHEMA,
        embedding_dimensions=s.embedding_dimensions,
        legacy_key_format=False,
    )


def clear_index(topic: str, settings: Settings | None = None) -> None:
    """Drop the Redis vector index and documents for a given topic.

    Args:
        topic: Topic name whose index should be removed.
        settings: Optional Settings instance.
    """
    s = settings or get_settings()
    index_name = get_index_name(topic, s)
    client = get_redis_client(s.redis_url)
    try:
        client.execute_command("FT.DROPINDEX", index_name, "DD")
        logger.info("Successfully dropped Redis index '%s'.", index_name)
    except ResponseError as exc:
        err_msg = str(exc).lower()
        ignore = ["no such index", "unknown index", "not found", "search_index_not_found"]
        if not any(token in err_msg for token in ignore):
            logger.warning("Error dropping Redis index '%s': %s", index_name, exc)
            raise
    except RedisConnectionError as exc:
        logger.warning("Redis connection error when clearing index '%s': %s", index_name, exc)
    finally:
        try:
            client.close()
        except Exception:
            pass


def clear_vectorstore(vectorstore: RedisVectorStore | None) -> None:
    """Safely clear an active RedisVectorStore instance."""
    if vectorstore is None:
        return
    idx = getattr(vectorstore, "index", None)
    if idx is None:
        return
    try:
        idx.clear()
    except ResponseError as exc:
        err_msg = str(exc).lower()
        ignore = ["no such index", "unknown index", "not found", "search_index_not_found"]
        if not any(token in err_msg for token in ignore):
            logger.warning("Error clearing vectorstore index: %s", exc)
            raise
    except Exception as exc:
        logger.debug("Non-critical error clearing vectorstore: %s", exc)


def create_vector_store(
    documents: List[Document],
    topic: str,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> RedisVectorStore:
    """Create and index documents into a fresh Redis vector store.

    Args:
        documents: Documents to be embedded and stored.
        topic: Topic string to partition the index.
        embeddings: LangChain Embeddings instance. Defaults to Google Generative AI.
        settings: Optional Settings instance.

    Returns:
        Populated RedisVectorStore ready for retrieval.
    """
    if not documents:
        raise ValueError("Cannot create a vector store with an empty document list.")

    s = settings or get_settings()

    # Drop existing index for this topic to ensure fresh reload
    clear_index(topic, s)

    # Normalize documents
    cleaned = clean_documents(documents)

    if embeddings is None:
        from researchmate.rag import get_embeddings
        embeddings = get_embeddings(s)

    config = get_redis_config(topic, s)

    logger.info("Indexing %d documents into Redis index '%s'...", len(cleaned), config.index_name)

    return RedisVectorStore.from_documents(
        documents=cleaned,
        embedding=embeddings,
        redis_url=s.redis_url,
        config=config,
    )
