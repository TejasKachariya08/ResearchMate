"""arXiv paper fetching module for ResearchMate.

Provides functions to search and download academic papers from arXiv,
parse abstracts and metadata, and handle rate-limiting and errors.
"""

from __future__ import annotations

import logging
import time
from typing import List

import arxiv
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def fetch_arxiv_papers(
    topic_query: str,
    num_papers: int = 10,
    delay_seconds: float = 2.0,
    max_retries: int = 4,
) -> List[Document]:
    """Fetch scientific papers from arXiv matching a search query.

    Args:
        topic_query: Search term or scientific topic (e.g. 'Reinforcement Learning').
        num_papers: Number of papers to fetch (clamped between 1 and 50).
        delay_seconds: Delay in seconds between API requests to prevent 429 errors.
        max_retries: Maximum number of attempts in case of rate limits or transient errors.

    Returns:
        List of LangChain Document objects with normalized metadata.
    """
    clean_query = topic_query.strip()
    if not clean_query:
        return []

    clamped_papers = max(1, min(int(num_papers), 50))

    search = arxiv.Search(
        query=clean_query,
        max_results=clamped_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    client = arxiv.Client(
        page_size=min(clamped_papers, 10),
        delay_seconds=delay_seconds,
        num_retries=max_retries,
    )

    documents: List[Document] = []
    retry_count = 0
    backoff = 2.0

    while retry_count < max_retries:
        try:
            logger.info("Fetching %d papers from arXiv for '%s' (attempt %d)...", clamped_papers, clean_query, retry_count + 1)
            for result in client.results(search):
                author_names = ", ".join(author.name for author in result.authors) if result.authors else "Unknown"
                category = str(result.primary_category) if result.primary_category else ""
                entry_url = str(result.entry_id) if result.entry_id else ""
                published_date = result.published.strftime("%Y-%m-%d") if getattr(result, "published", None) else ""

                # Build Document with standardized metadata
                doc = Document(
                    page_content=result.summary.strip(),
                    metadata={
                        "title": result.title.strip().replace("\n", " "),
                        "Title": result.title.strip().replace("\n", " "),
                        "authors": author_names,
                        "Authors": author_names,
                        "category": category,
                        "primary_category": category,
                        "links": entry_url,
                        "published": published_date,
                    },
                )
                documents.append(doc)

            if documents:
                logger.info("Successfully fetched %d papers for '%s'.", len(documents), clean_query)
                return documents

            logger.warning("No papers found on arXiv matching '%s'.", clean_query)
            return []

        except arxiv.HTTPError as exc:
            retry_count += 1
            wait_time = backoff * retry_count
            logger.warning("arXiv HTTP error: %s. Retrying in %.1f seconds...", exc, wait_time)
            time.sleep(wait_time)

        except Exception as exc:
            retry_count += 1
            wait_time = backoff * retry_count
            logger.warning("Error fetching papers: %s. Retrying in %.1f seconds...", exc, wait_time)
            time.sleep(wait_time)

    logger.error("Failed to fetch arXiv papers after %d retries.", max_retries)
    return []
