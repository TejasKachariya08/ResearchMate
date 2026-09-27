"""ResearchMate: AI-Powered Academic Research Assistant.

Grounded Q&A over scientific papers using Redis Vector Search and Google Gemini.
"""

from researchmate.config import Settings, get_settings
from researchmate.arxiv import fetch_arxiv_papers
from researchmate.vector_store import create_vector_store, clear_vectorstore, clear_index
from researchmate.rag import get_llm, get_embeddings, build_rag_chain

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "get_settings",
    "fetch_arxiv_papers",
    "create_vector_store",
    "clear_vectorstore",
    "clear_index",
    "get_llm",
    "get_embeddings",
    "build_rag_chain",
    "__version__",
]
