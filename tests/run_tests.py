"""ResearchMate Standalone Test Suite Runner.

Executes all unit tests using Python's standard unittest library.
Zero third-party test dependencies required.

Usage:
    python tests/run_tests.py
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src directory is in sys.path
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import modules under test
from langchain_core.documents import Document
from redis.exceptions import ResponseError

from researchmate.config import (
    Settings,
    ensure_google_api_key,
    get_index_name,
    get_index_prefix,
    sanitize_topic_key,
)
from researchmate.arxiv import fetch_arxiv_papers
from researchmate.vector_store import (
    clean_documents,
    clear_index,
    clear_vectorstore,
    get_redis_config,
    create_vector_store,
)
from researchmate.rag import (
    format_retrieved_docs,
    get_research_prompt,
    build_rag_chain,
)
from researchmate.stats import (
    build_attribute_rows,
    build_index_rows,
    build_stats_rows,
    get_index_info,
)


class TestConfig(unittest.TestCase):
    """Test configuration loading and key sanitization."""

    def test_settings_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            s = Settings.from_env()
            self.assertEqual(s.gemini_chat_model, "models/gemini-3.8-flash")
            self.assertEqual(s.gemini_embedding_model, "models/gemini-embedding-2")
            self.assertEqual(s.embedding_dimensions, 3072)
            self.assertEqual(s.redis_url, "redis://localhost:6379")
            self.assertEqual(s.redis_index_basename, "researchmate")

    def test_settings_custom(self):
        env_vars = {
            "GOOGLE_API_KEY": "custom_api_key_xyz",
            "GEMINI_CHAT_MODEL": "gemini-3.6-flash",
            "GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
            "EMBEDDING_DIMENSIONS": "1536",
            "REDIS_URL": "redis://my-redis:6380",
            "REDIS_INDEX_BASENAME": "custom_mate",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            s = Settings.from_env()
            self.assertEqual(s.google_api_key, "custom_api_key_xyz")
            self.assertEqual(s.gemini_chat_model, "models/gemini-3.6-flash")
            self.assertEqual(s.gemini_embedding_model, "models/gemini-embedding-2")
            self.assertEqual(s.embedding_dimensions, 1536)
            self.assertEqual(s.redis_url, "redis://my-redis:6380")
            self.assertEqual(s.redis_index_basename, "custom_mate")

    def test_ensure_google_api_key_missing(self):
        s = Settings(google_api_key="")
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}, clear=True):
            with self.assertRaises(ValueError):
                ensure_google_api_key(s)

    def test_sanitize_topic_key(self):
        self.assertEqual(sanitize_topic_key("Quantum Computing & LLMs!"), "quantum_computing_llms")
        self.assertEqual(sanitize_topic_key("   Reinforcement   Learning   "), "reinforcement_learning")
        self.assertEqual(sanitize_topic_key("---"), "general")

    def test_index_and_prefix_generation(self):
        s = Settings(redis_index_basename="researchmate")
        self.assertEqual(get_index_name("Diffusion Models", s), "researchmate_diffusion_models")
        self.assertEqual(get_index_prefix("Diffusion Models", s), "researchmate:doc:diffusion_models:")


class TestArXiv(unittest.TestCase):
    """Test arXiv fetching and paper parsing."""

    def test_empty_query(self):
        self.assertEqual(fetch_arxiv_papers(""), [])
        self.assertEqual(fetch_arxiv_papers("   "), [])

    @patch("arxiv.Client")
    def test_fetch_papers_success(self, mock_client_cls):
        author = MagicMock()
        author.name = "Geoffrey Hinton"

        res = MagicMock()
        res.title = "Deep Learning Fundamentals\nSecond Edition"
        res.summary = "A comprehensive survey of deep neural networks."
        res.authors = [author]
        res.primary_category = "cs.LG"
        res.entry_id = "http://arxiv.org/abs/2101.00001"
        res.published = datetime(2021, 5, 20)

        mock_client = MagicMock()
        mock_client.results.return_value = [res]
        mock_client_cls.return_value = mock_client

        docs = fetch_arxiv_papers("deep learning", num_papers=1, delay_seconds=0)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["title"], "Deep Learning Fundamentals Second Edition")
        self.assertEqual(docs[0].metadata["authors"], "Geoffrey Hinton")
        self.assertEqual(docs[0].metadata["category"], "cs.LG")
        self.assertEqual(docs[0].metadata["links"], "http://arxiv.org/abs/2101.00001")


class TestVectorStore(unittest.TestCase):
    """Test Redis vector store management and document formatting."""

    def setUp(self):
        self.sample_docs = [
            Document(
                page_content="Paper on Transformer architectures.",
                metadata={
                    "Title": "Attention Is All You Need",
                    "Authors": "Vaswani et al.",
                    "primary_category": "cs.CL",
                    "links": "https://arxiv.org/abs/1706.03762",
                },
            )
        ]

    def test_clean_documents(self):
        cleaned = clean_documents(self.sample_docs)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0].metadata["title"], "Attention Is All You Need")
        self.assertEqual(cleaned[0].metadata["authors"], "Vaswani et al.")
        self.assertEqual(cleaned[0].metadata["category"], "cs.CL")

    def test_get_redis_config(self):
        s = Settings(embedding_dimensions=3072, redis_index_basename="researchmate")
        cfg = get_redis_config("Transformers", s)
        self.assertEqual(cfg.index_name, "researchmate_transformers")
        self.assertEqual(cfg.key_prefix, "researchmate:doc:transformers:")
        self.assertEqual(cfg.embedding_dimensions, 3072)

    @patch("researchmate.vector_store.get_redis_client")
    def test_clear_index_suppresses_unknown_index(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.execute_command.side_effect = ResponseError("Unknown Index name")
        mock_get_client.return_value = mock_client

        # Should not raise exception
        clear_index("nonexistent_topic")
        mock_client.execute_command.assert_called_once()
        mock_client.close.assert_called_once()

    def test_clear_vectorstore_safe(self):
        clear_vectorstore(None)
        mock_vs = MagicMock()
        mock_vs.index.clear.side_effect = ResponseError("no such index")
        clear_vectorstore(mock_vs)  # Should gracefully catch error

    @patch("researchmate.vector_store.clear_index")
    @patch("researchmate.vector_store.RedisVectorStore.from_documents")
    def test_create_vector_store(self, mock_from_docs, mock_clear):
        mock_from_docs.return_value = "mock_store"
        mock_emb = MagicMock()

        store = create_vector_store(self.sample_docs, "Transformers", embeddings=mock_emb)
        self.assertEqual(store, "mock_store")
        mock_clear.assert_called_once()
        self.assertEqual(mock_clear.call_args[0][0], "Transformers")
        mock_from_docs.assert_called_once()


class TestRAG(unittest.TestCase):
    """Test RAG prompt templating and document context formatting."""

    def test_prompt_template(self):
        p = get_research_prompt()
        res = p.format(context="Test context", question="Test question")
        self.assertIn("Test context", res)
        self.assertIn("Test question", res)
        self.assertIn("ResearchMate", res)

    def test_format_retrieved_docs(self):
        docs = [
            Document(
                page_content="Paper 1 content",
                metadata={"title": "Paper 1", "authors": "Author A"},
            ),
            Document(
                page_content="Paper 2 content",
                metadata={"title": "Paper 2", "authors": "Author B"},
            ),
        ]
        text = format_retrieved_docs(docs)
        self.assertIn("[1] Title: Paper 1", text)
        self.assertIn("Paper 1 content", text)
        self.assertIn("[2] Title: Paper 2", text)
        self.assertIn("Paper 2 content", text)

    def test_build_rag_chain(self):
        mock_llm = MagicMock()
        mock_vs = MagicMock()
        chain = build_rag_chain(mock_llm, mock_vs, k=4)
        self.assertIsNotNone(chain)
        mock_vs.as_retriever.assert_called_once_with(
            search_kwargs={"k": 4},
            search_type="similarity",
        )


class TestStats(unittest.TestCase):
    """Test Redis stats parsing and schema rows construction."""

    def test_get_index_info(self):
        mock_redis = MagicMock()
        mock_redis.ft.return_value.info.return_value = {"index_name": "test_idx"}
        info = get_index_info("test_idx", redis_client=mock_redis)
        self.assertEqual(info["index_name"], "test_idx")

    def test_build_index_rows(self):
        info = {
            "index_name": "researchmate_rl",
            "index_definition": ["key_type", "HASH", "prefixes", ["researchmate:doc:rl:"]],
            "index_options": [],
            "indexing": 0,
        }
        rows = build_index_rows(info)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Index Name"], "researchmate_rl")
        self.assertEqual(rows[0]["Storage Type"], "HASH")

    def test_build_attribute_rows(self):
        info = {
            "attributes": [
                ["identifier", "embedding", "attribute", "embedding", "type", "VECTOR", "algorithm", "FLAT"]
            ]
        }
        rows = build_attribute_rows(info)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Identifier"], "embedding")
        self.assertEqual(rows[0]["Type"], "VECTOR")

    def test_build_stats_rows(self):
        info = {"num_docs": 10, "num_records": 10, "vector_index_sz_mb": 0.5}
        rows = build_stats_rows(info)
        stats = {r["Metric"]: r["Value"] for r in rows}
        self.assertEqual(stats["num_docs"], 10)
        self.assertEqual(stats["vector_index_sz_mb"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
