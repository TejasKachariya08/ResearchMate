"""Tests for arXiv fetching module in ResearchMate."""

from unittest.mock import MagicMock, patch
from datetime import datetime
from researchmate.arxiv import fetch_arxiv_papers


def test_fetch_arxiv_empty_query():
    assert fetch_arxiv_papers("") == []
    assert fetch_arxiv_papers("   ") == []


def test_fetch_arxiv_papers_success():
    mock_author1 = MagicMock()
    mock_author1.name = "Alice Smith"
    mock_author2 = MagicMock()
    mock_author2.name = "Bob Jones"

    mock_result = MagicMock()
    mock_result.title = "Sample Paper Title\nWith Newline"
    mock_result.summary = "This is a summary of the paper."
    mock_result.authors = [mock_author1, mock_author2]
    mock_result.primary_category = "cs.AI"
    mock_result.entry_id = "http://arxiv.org/abs/2301.00001v1"
    mock_result.published = datetime(2023, 1, 15)

    with patch("arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_cls.return_value = mock_client

        docs = fetch_arxiv_papers("quantum computing", num_papers=1, delay_seconds=0)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.page_content == "This is a summary of the paper."
        assert doc.metadata["title"] == "Sample Paper Title With Newline"
        assert doc.metadata["authors"] == "Alice Smith, Bob Jones"
        assert doc.metadata["category"] == "cs.AI"
        assert doc.metadata["links"] == "http://arxiv.org/abs/2301.00001v1"
        assert doc.metadata["published"] == "2023-01-15"


def test_fetch_arxiv_papers_retry_on_error():
    with patch("arxiv.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.results.side_effect = [Exception("Temporary arXiv error"), []]
        mock_client_cls.return_value = mock_client

        with patch("time.sleep", return_value=None):
            docs = fetch_arxiv_papers("test", num_papers=2, max_retries=2, delay_seconds=0)
            assert docs == []
