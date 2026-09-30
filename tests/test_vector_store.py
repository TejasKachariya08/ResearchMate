"""Tests for Redis Vector Store management in ResearchMate."""

import pytest
from unittest.mock import MagicMock, patch
from redis.exceptions import ResponseError
from langchain_core.documents import Document

from researchmate.config import Settings
from researchmate.vector_store import (
    clean_documents,
    clear_index,
    clear_vectorstore,
    get_redis_config,
    create_vector_store,
)


def test_clean_documents_standardizes_metadata(sample_raw_documents):
    cleaned = clean_documents(sample_raw_documents)
    assert len(cleaned) == 2

    assert cleaned[0].metadata["title"] == "Training language models to follow instructions"
    assert cleaned[0].metadata["authors"] == "Long Ouyang, Jeff Wu, Xu Jiang"
    assert cleaned[0].metadata["category"] == "cs.CL"
    assert cleaned[0].metadata["links"] == "https://arxiv.org/abs/2203.02155"

    assert cleaned[1].metadata["title"] == "Direct Preference Optimization"
    assert cleaned[1].metadata["authors"] == "Rafael Rafailov, Archit Sharma"


def test_get_redis_config_topic_scoping():
    settings = Settings(
        embedding_dimensions=3072,
        redis_index_basename="researchmate",
        redis_url="redis://localhost:6379",
    )
    config = get_redis_config("Diffusion Models", settings=settings)

    assert config.index_name == "researchmate_diffusion_models"
    assert config.key_prefix == "researchmate:doc:diffusion_models:"
    assert config.embedding_dimensions == 3072
    assert config.storage_type == "hash"
    assert any(f["name"] == "category" for f in config.metadata_schema)


def test_clear_index_suppresses_missing_index_error():
    fake_client = MagicMock()
    fake_client.execute_command.side_effect = ResponseError("Unknown Index name")

    with patch("researchmate.vector_store.get_redis_client", return_value=fake_client):
        # Should not raise
        clear_index("Nonexistent Topic")

    fake_client.execute_command.assert_called_once_with("FT.DROPINDEX", "researchmate_nonexistent_topic", "DD")
    fake_client.close.assert_called_once()


def test_clear_vectorstore_handles_none_and_exceptions():
    clear_vectorstore(None)  # Should gracefully do nothing

    mock_store = MagicMock()
    mock_store.index.clear.side_effect = ResponseError("search_index_not_found")
    clear_vectorstore(mock_store)  # Should catch and ignore


def test_create_vector_store_empty_raises():
    with pytest.raises(ValueError, match="Cannot create a vector store with an empty"):
        create_vector_store([], "Empty Topic")


def test_create_vector_store_invokes_indexing(sample_raw_documents):
    mock_embeddings = MagicMock()
    with patch("researchmate.vector_store.clear_index") as mock_clear, \
         patch("researchmate.vector_store.RedisVectorStore.from_documents", return_value="mock_vstore") as mock_from_docs:

        store = create_vector_store(sample_raw_documents, "Topic A", embeddings=mock_embeddings)

        assert store == "mock_vstore"
        mock_clear.assert_called_once()
        assert mock_clear.call_args[0][0] == "Topic A"
        mock_from_docs.assert_called_once()
