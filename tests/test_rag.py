"""Tests for RAG pipeline and prompt generation in ResearchMate."""

from unittest.mock import MagicMock
from langchain_core.documents import Document
from researchmate.rag import (
    format_retrieved_docs,
    get_research_prompt,
    build_rag_chain,
)


def test_research_prompt_template():
    prompt = get_research_prompt()
    rendered = prompt.format(
        context="Paper 1 findings",
        question="What is the key algorithm?",
    )

    assert "Paper 1 findings" in rendered
    assert "What is the key algorithm?" in rendered
    assert "You are ResearchMate" in rendered
    assert "Answer in Markdown:" in rendered


def test_format_retrieved_docs():
    docs = [
        Document(
            page_content="Text from paper 1.",
            metadata={"title": "Paper 1", "authors": "Author A"},
        ),
        Document(
            page_content="Text from paper 2.",
            metadata={"title": "Paper 2", "authors": "Author B"},
        ),
    ]

    formatted = format_retrieved_docs(docs)
    assert "[1] Title: Paper 1" in formatted
    assert "Authors: Author A" in formatted
    assert "Text from paper 1." in formatted
    assert "[2] Title: Paper 2" in formatted
    assert "Authors: Author B" in formatted


def test_build_rag_chain():
    mock_llm = MagicMock()
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    chain = build_rag_chain(
        llm=mock_llm,
        vector_store=mock_vectorstore,
        k=3,
        search_type="similarity",
    )

    assert chain is not None
    mock_vectorstore.as_retriever.assert_called_once_with(
        search_kwargs={"k": 3},
        search_type="similarity",
    )
