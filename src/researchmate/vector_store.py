from __future__ import annotations
from langchain_core.documents import Document
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis
from researchmate.config import get_settings

def get_redis_client() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url, decode_responses=False)

def clear_index(topic: str) -> None:
    settings = get_settings()
    client = get_redis_client()
    client.execute_command("FT.DROPINDEX", settings.get_index_name(topic), "DD")
    client.close()

def create_vector_store(documents: list[Document], topic: str) -> RedisVectorStore:
    clear_index(topic)
    settings = get_settings()
    config = RedisConfig(
        index_name=settings.get_index_name(topic),
        key_prefix=settings.get_index_prefix(topic),
        redis_url=settings.redis_url,
        storage_type="hash",
    )
    return RedisVectorStore.from_documents(documents, None, redis_url=settings.redis_url, config=config)
