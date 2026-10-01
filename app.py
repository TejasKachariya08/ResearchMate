"""ResearchMate: Academic Research Synthesis & Literature Q&A.

Grounded on topic-scoped Redis Vector Search and Google Gemini.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import defaultdict

# Ensure src is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
from researchmate.config import get_settings, get_index_name
from researchmate.arxiv import fetch_arxiv_papers
from researchmate.vector_store import create_vector_store, clear_vectorstore
from researchmate.rag import get_llm, build_rag_chain, get_research_prompt
from researchmate.utils import check_redis_connection

# Page configuration
st.set_page_config(
    page_title="ResearchMate | Academic AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Automated Model & Retrieval Settings (Zero manual configuration needed)
AUTO_MAX_TOKENS = 850
AUTO_TEMPERATURE = 0.25
AUTO_CONTEXT_DOCS = 4


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables with clean defaults."""
    defaults = {
        "messages": [],
        "vector_store": None,
        "loaded_papers": [],
        "loaded_topic": "",
        "loaded_num_papers": 0,
        "active_index_name": "",
        "cached_context": [],
        "last_answer": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def handle_clear_chat() -> None:
    """Clear conversation history while retaining loaded index and papers."""
    st.session_state["messages"] = []
    st.session_state["cached_context"] = []
    st.session_state["last_answer"] = ""


def handle_reset_all() -> None:
    """Clear conversation, loaded papers, and drop active vector store."""
    if st.session_state.get("vector_store") is not None:
        clear_vectorstore(st.session_state["vector_store"])
    st.session_state["vector_store"] = None
    st.session_state["loaded_papers"] = []
    st.session_state["loaded_topic"] = ""
    st.session_state["loaded_num_papers"] = 0
    st.session_state["active_index_name"] = ""
    handle_clear_chat()


initialize_session_state()
settings = get_settings()

# -----------------------------------------------------------------------------
# Deep Obsidian Black Editorial Theme (Authentic, Non-AI Aesthetic)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400;1,6..72,500&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Dark Background & Typography */
    .stApp {
        background-color: #090a0f !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.25rem !important;
        padding-top: 5.75rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1120px !important;
    }

    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.025em !important;
    }

    p, span, label, div {
        color: #cbd5e1;
    }

    /* Top Divider */
    hr {
        border-color: #1e222d !important;
        margin: 0.65rem  1.5rem 0 !important;
    }

    /* Navigation Bar */
    .nav-brand-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.42rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.025em;
        line-height: 1.15;
    }

    .nav-brand-subtitle {
        font-size: 0.74rem;
        color: #94a3b8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Containers and Cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
        border: 1px solid #1e222d !important;
        background-color: #111318 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) !important;
    }

  

    /* Button Styling */
    .stButton button {
        border-radius: 6px !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        padding: 0.42rem 0.95rem !important;
        border: 1px solid #262a36 !important;
        background-color: #3b82f6 !important;
        color: #e2e8f0 !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton button:hover {
        background-color: #1d212c !important;
        border-color: #3b4255 !important;
        color: #ffffff !important;
    }

    .stButton button:active {
        background-color: #111318 !important;
    }

    /* Primary Accent Button for Indexing */
    div[data-testid="stColumn"]:last-child .stButton button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stColumn"]:last-child .stButton button:hover {
        background-color: #1d4ed8 !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px solid #1e222d !important;
        border-radius: 8px !important;
        background-color: #111318 !important;
        margin-bottom: 0.65rem !important;
    }

    [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #93c5fd !important;
    }

    /* Paper Abstract Container */
    .paper-abstract-box {
        font-family: 'Newsreader', 'Charter', Georgia, serif;
        font-size: 1.02rem;
        line-height: 1.65;
        color: #e2e8f0;
        background-color: #0b0c10;
        border-left: 3px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 6px 6px 0;
        margin-top: 10px;
        border-top: 1px solid #171922;
        border-right: 1px solid #171922;
        border-bottom: 1px solid #171922;
    }

    /* Active Topic Pill */
    .status-topic-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background-color: #12141c;
        border: 1px solid #202430;
        color: #cbd5e1;
        font-size: 0.825rem;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .status-topic-pill strong {
        color: #ffffff;
    }

    .status-topic-pill code {
        background-color: #0b0c10;
        color: #93c5fd;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #1e2433;
    }



    .stChatInput {
        width: 63rem !important;
        margin: 0 auto !important;
    }

    .st-emotion-cache-1mplmdc {
        border-radius: 10rem;
    }
    .st-emotion-cache-1126q7t {
        border-radius: 10rem;
    }
    .st-emotion-cache-18hdgyo {
        border-radius: 10rem;
    }
    .nav-container {
       border 10px solid #1e222d
    /* Streamlit Default Header Adjustment */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        z-index: 1000000 !important;
    }

    
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        pointer-events: auto !important;
    }

   
    /* Fixed Navigation Bar Container */
    div[data-testid="stLayoutWrapper"]:has(.nav-brand-title),
    div:has(> [data-testid="stHorizontalBlock"] .nav-brand-title),
    .st-key-navbar {
        position: fixed !important;
        top: 0 !important;
        left: 5rem !important;
        right: 0 !important;
        width: 85% !important;
        z-index: 99999 !important;
        background-color: rgba(9, 10, 15, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 0.6rem 0 !important;
    }

    /* Inner Row Centering & Max Width */
    div[data-testid="stLayoutWrapper"]:has(.nav-brand-title) [data-testid="stHorizontalBlock"],
    div:has(> [data-testid="stHorizontalBlock"] .nav-brand-title) [data-testid="stHorizontalBlock"],
    .st-key-navbar [data-testid="stHorizontalBlock"] {
        max-width: 1120px !important;
        margin: 0 auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        align-items: center !important;
    }

    /* Navbar Action Buttons */
    div[data-testid="stLayoutWrapper"]:has(.nav-brand-title) .stButton button,
    div:has(> [data-testid="stHorizontalBlock"] .nav-brand-title) .stButton button,
    .st-key-navbar .stButton button {
        background-color: #14171f !important;
        color: #e2e8f0 !important;
        border: 1px solid #262a36 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stLayoutWrapper"]:has(.nav-brand-title) .stButton button:hover,
    div:has(> [data-testid="stHorizontalBlock"] .nav-brand-title) .stButton button:hover,
    .st-key-navbar .stButton button:hover {
        background-color: #1d212c !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Top Navigation Bar with Two Session Control Buttons
# Top Navigation Bar with Two Session Control Buttons (Fixed at Top)
# -----------------------------------------------------------------------------
nav_col_brand, nav_col_clear, nav_col_reset,nav_col_container = st.columns([5.5, 1.25, 1.25], vertical_alignment="center")
with nav_col_container:
    st.markdown(
        """
        <div class="nav-container">
        """,
        unsafe_allow_html=True,
    )
    
with st.container(key="navbar"):
    nav_col_brand, nav_col_clear, nav_col_reset = st.columns([5.5, 1.25, 1.25], vertical_alignment="center")

    with nav_col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; padding: 10px 10px 10px 10px;">
            <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0;">
                <div style="background: #1e222e; border: 1px solid #333846; color: #ffffff; font-weight: 700; font-size: 13px; letter-spacing: 0.08em; padding: 6px 10px; border-radius: 6px;">RM</div>
                <div>
                    <div class="nav-brand-title">ResearchMate</div>
                    <div class="nav-brand-subtitle">Academic Literature Retrieval & Grounded Synthesis</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col_clear:
        st.button(
            "Clear Chat",
            on_click=handle_clear_chat,
            use_container_width=True,
            help="Clear conversation messages while keeping loaded papers and Redis index",
        )

    with nav_col_reset:
        st.button(
            "Reset Topic",
            on_click=handle_reset_all,
            use_container_width=True,
            help="Drop current index, clear papers, and start a fresh research query",
        )
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr />", unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# Topic Ingestion Box
# -----------------------------------------------------------------------------
with st.container(border=True):
    col_topic, col_papers, col_btn = st.columns([3.4, 1.1, 1.2], vertical_alignment="bottom")

    with col_topic:
        topic_input = st.text_input(
            "Research Domain or arXiv Query",
            placeholder="e.g. Constitutional AI, Direct Preference Optimization, Quantum Graph Neural Networks...",
            help="Enter scientific subject, preprint keywords, or methodologies.",
        )

    with col_papers:
        paper_count = st.number_input(
            "Paper Count",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            help="Number of most relevant peer-reviewed/preprint papers to index.",
        )

    with col_btn:
        ingest_clicked = st.button("Index Papers", use_container_width=True)


# -----------------------------------------------------------------------------
# Ingestion Action
# -----------------------------------------------------------------------------
if ingest_clicked:
    cleaned_topic = topic_input.strip()
    if not cleaned_topic:
        st.warning("Please enter a research topic or search query before indexing.")
    else:
        with st.status("Querying and indexing research papers...", expanded=True) as status:
            st.write(f"1. Querying arXiv repository for **'{cleaned_topic}'**...")
            docs = fetch_arxiv_papers(cleaned_topic, num_papers=paper_count)

            if not docs:
                status.update(label="No papers found or error contacting arXiv.", state="error")
                st.error("No papers found on arXiv for this query. Try a different topic.")
            else:
                st.write(f"2. Found {len(docs)} papers. Generating embeddings with Gemini...")
                st.write("3. Storing vector embeddings in Redis...")
                try:
                    vstore = create_vector_store(docs, cleaned_topic, settings=settings)
                    st.session_state["vector_store"] = vstore
                    st.session_state["loaded_papers"] = docs
                    st.session_state["loaded_topic"] = cleaned_topic
                    st.session_state["loaded_num_papers"] = len(docs)
                    st.session_state["active_index_name"] = get_index_name(cleaned_topic, settings)
                    status.update(label=f"Indexed {len(docs)} papers into Redis successfully!", state="complete")
                    st.toast("Corpus ready! Review summaries and ask questions below.", icon="📘")
                except Exception as exc:
                    status.update(label="Failed to index papers in Redis.", state="error")
                    st.error(f"Error during Redis indexing: {exc}")


# Active Topic Info & Warning
if st.session_state.get("loaded_topic"):
    st.markdown(
        f"""
        <div class="status-topic-pill">
            <span>📚 Active Corpus: <strong>{st.session_state['loaded_topic']}</strong></span>
            <span>•</span>
            <span>Indexed Papers: <strong>{st.session_state['loaded_num_papers']}</strong></span>
            <span>•</span>
            <span>Redis Index: <code>{st.session_state['active_index_name']}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if topic_input and st.session_state.get("loaded_topic") and topic_input.strip() != st.session_state["loaded_topic"]:
    st.warning("Notice: The search input changed. Click 'Index Papers' to rebuild the index for this topic.")


# -----------------------------------------------------------------------------
# Section: Paper-Wise Summaries
# -----------------------------------------------------------------------------
if st.session_state.get("loaded_papers"):
    st.divider()
    st.subheader(f"📑 Paper Summaries & Abstracts ({len(st.session_state['loaded_papers'])} Papers)")
    st.markdown(
        f"Review the abstracts and methodologies retrieved for **{st.session_state['loaded_topic']}**. "
        "Use the chat section below to query and compare these papers."
    )

    for idx, doc in enumerate(st.session_state["loaded_papers"], 1):
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("Title") or f"Paper #{idx}"
        authors = meta.get("authors") or meta.get("Authors") or "Unknown Authors"
        category = meta.get("category") or meta.get("primary_category") or "General"
        links = meta.get("links") or ""
        published = meta.get("published") or ""

        with st.expander(f"📄 **{idx}. {title}**", expanded=(idx <= 2)):
            col_info, col_link = st.columns([4, 1])
            with col_info:
                st.markdown(f"**👥 Authors:** {authors}")
                meta_line = []
                if category:
                    meta_line.append(f"🏷️ **Category:** `{category}`")
                if published:
                    meta_line.append(f"📅 **Published:** {published}")
                if meta_line:
                    st.caption(" | ".join(meta_line))
            with col_link:
                if links:
                    st.link_button("🔗 arXiv Link", links, use_container_width=True)

            st.markdown("**📝 Abstract / Summary:**")
            st.markdown(
                f"""
                <div class="paper-abstract-box">
                    {doc.page_content.strip()}
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# Section: Research Q&A Chat
# -----------------------------------------------------------------------------
if st.session_state.get("loaded_papers"):
    st.divider()
    st.subheader("💬 Ask Questions About These Papers")

    # Display previous conversation messages
    for message in st.session_state["messages"]:
        role = message["role"]
        avatar = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

    # Only activate chat handling if papers have been loaded into vector store
    if st.session_state.get("vector_store") is not None:
        # Initialize RAG Pipeline automatically with optimal settings
        try:
            llm = get_llm(
                max_tokens=AUTO_MAX_TOKENS,
                temperature=AUTO_TEMPERATURE,
                settings=settings,
            )
            rag_chain = build_rag_chain(
                llm=llm,
                vector_store=st.session_state["vector_store"],
                prompt=get_research_prompt(),
                k=AUTO_CONTEXT_DOCS,
            )
        except Exception as e:
            st.error(f"Failed to initialize language model: {e}")
            st.stop()

        # Chat Input for active session
        if user_query := st.chat_input("Ask a question about the loaded papers..."):
            # Append & display user message
            st.session_state["messages"].append({"role": "user", "content": user_query})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_query)

            # Generate grounded response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching papers and synthesizing answer..."):
                    try:
                        response_dict = rag_chain.invoke(user_query)
                        answer_text = response_dict.get("answer", "")
                        context_docs = response_dict.get("context", [])

                        st.markdown(answer_text)

                        # Expandable citations and context view
                        if context_docs:
                            with st.expander("📚 Referenced Sources & Context Chunks"):
                                grouped_docs = defaultdict(list)
                                for doc in context_docs:
                                    t = doc.metadata.get("title") or "Unknown Paper"
                                    grouped_docs[t].append(doc)

                                for c_idx, (c_title, doc_list) in enumerate(grouped_docs.items(), 1):
                                    first_doc = doc_list[0]
                                    c_authors = first_doc.metadata.get("authors", "Unknown")
                                    c_link = first_doc.metadata.get("links", "")

                                    link_md = f"([arXiv Link]({c_link}))" if c_link else ""
                                    st.markdown(f"**{c_idx}. {c_title}** {link_md}")
                                    st.caption(f"Authors: {c_authors}")

                                    for chunk_i, chunk in enumerate(doc_list, 1):
                                        st.markdown(f"> *Excerpt {chunk_i}:* {chunk.page_content}")
                                    st.divider()

                        # Save assistant response to session
                        st.session_state["messages"].append({"role": "assistant", "content": answer_text})

                    except Exception as exc:
                        err_msg = f"Failed to generate response: {exc}"
                        st.error(err_msg)
                        st.session_state["messages"].append({"role": "assistant", "content": err_msg})
    else:
        # Clean disabled chat input when no topic has been ingested yet
        st.chat_input("Enter a topic above and click 'Index Papers' to start asking questions...", disabled=True)
