"""arXiv paper fetching module for ResearchMate."""
from __future__ import annotations
import arxiv
from typing import List
from langchain_core.documents import Document

def fetch_arxiv_papers(topic_query: str, num_papers: int = 10) -> List[Document]:
    search = arxiv.Search(
        query=topic_query,
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    documents = []
    for result in client.results(search):
        documents.append(
            Document(
                page_content=result.summary,
                metadata={
                    "title": result.title,
                    "authors": ", ".join(a.name for a in result.authors),
                    "primary_category": str(result.primary_category) if result.primary_category else "",
                    "links": str(result.entry_id),
                },
            )
        )
    return documents
