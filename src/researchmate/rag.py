"""Retrieval-Augmented Generation (RAG) module for ResearchMate.

Constructs LangChain LCEL pipelines combining Google Gemini models,
Redis vector search retrieval, and grounded citation prompts.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_redis import RedisVectorStore

from researchmate.config import Settings, ensure_google_api_key, get_settings

logger = logging.getLogger(__name__)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create Google Generative AI embeddings client.

    Args:
        settings: Optional Settings instance.

    Returns:
        Configured GoogleGenerativeAIEmbeddings instance.
    """
    s = settings or get_settings()
    api_key = ensure_google_api_key(s)
    return GoogleGenerativeAIEmbeddings(
        model=s.gemini_embedding_model,
        google_api_key=api_key,
    )


def get_llm(
    max_tokens: int = 500,
    temperature: float = 0.5,
    settings: Settings | None = None,
    **kwargs,
) -> BaseChatModel:
    """Instantiate Google Gemini Chat model.

    Args:
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        settings: Optional Settings instance.
        kwargs: Additional parameters passed to ChatGoogleGenerativeAI.

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    s = settings or get_settings()
    api_key = ensure_google_api_key(s)

    model_name = s.gemini_chat_model
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


def get_research_prompt() -> PromptTemplate:
    """Construct the academic research grounding prompt template."""
    template = (
        "You are ResearchMate, an advanced academic research assistant.\n"
        "Answer the user's question accurately and objectively based ONLY on the provided research context from arXiv papers.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Instructions:\n"
        "1. Rely strictly on the facts directly mentioned in the Context. Do not speculate or extrapolate beyond what is stated.\n"
        "2. If the context does not contain sufficient information to answer the question, clearly state: "
        "'I don't have enough information from the loaded papers to answer this question.'\n"
        "3. Where relevant, reference specific paper titles and findings mentioned in the context.\n"
        "4. Keep your answer clear, organized, and academic.\n\n"
        "Answer in Markdown:"
    )
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def format_retrieved_docs(docs: List[Document]) -> str:
    """Format list of retrieved Document objects into a readable context string."""
    formatted_chunks: List[str] = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title") or doc.metadata.get("Title") or "Unknown Paper"
        authors = doc.metadata.get("authors") or doc.metadata.get("Authors") or "Unknown Authors"
        content = doc.page_content.strip()
        formatted_chunks.append(f"[{i}] Title: {title}\nAuthors: {authors}\nContent:\n{content}")
    return "\n\n---\n\n".join(formatted_chunks)


def build_rag_chain(
    llm: BaseChatModel,
    vector_store: RedisVectorStore,
    prompt: PromptTemplate | None = None,
    k: int = 4,
    search_type: str = "similarity",
) -> Runnable:
    """Construct an LCEL RAG pipeline that retrieves context and generates answers.

    Args:
        llm: Language model instance.
        vector_store: Populated RedisVectorStore.
        prompt: Optional custom PromptTemplate.
        k: Number of nearest neighbor chunks to retrieve.
        search_type: Retrieval strategy (e.g. 'similarity').

    Returns:
        Runnable that accepts a query string or dict and returns a dict with 'context' and 'answer'.
    """
    active_prompt = prompt or get_research_prompt()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k},
        search_type=search_type,
    )

    answer_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_retrieved_docs(x["context"])))
        | active_prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=answer_chain)

    return rag_chain

