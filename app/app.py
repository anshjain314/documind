"""
app/app.py

Streamlit UI for DocuMind. Shows the question, the final answer, and a
transparency panel revealing which path the agentic pipeline took -
this is what makes the "agentic" behavior visible and demoable.

Run from the project root:
    streamlit run app/app.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path so `graph` and `ingestion` package imports work
# regardless of how Streamlit invokes this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# On Streamlit Cloud, API keys are provided via st.secrets rather than a
# local .env file. Copy them into environment variables here, BEFORE
# importing the graph modules below (which read os.environ at import time).
# Locally, .env (loaded via dotenv inside the graph modules) still works fine.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets.toml present (e.g. running fully locally with .env) - fine

from graph.build_graph import compiled_graph
from ingestion.build_index import ingest_pdfs

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="centered")

st.title("📄 DocuMind")
st.caption("Agentic RAG assistant — ask questions about your uploaded documents")

# Keep a simple history in session state so the page doesn't reset each query
if "history" not in st.session_state:
    st.session_state.history = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# --- Sidebar: document upload ---
with st.sidebar:
    st.markdown("### 📤 Upload documents")
    uploaded_files = st.file_uploader(
        "Add PDF(s) to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    replace_existing = st.checkbox(
        "Replace existing documents",
        value=False,
        help="If checked, clears all previously ingested documents first. "
             "If unchecked, new documents are added alongside existing ones.",
    )

    if uploaded_files and st.button("Ingest uploaded PDFs", type="primary"):
        status_box = st.empty()
        log_lines = []

        def update_status(msg):
            log_lines.append(msg)
            status_box.text("\n".join(log_lines[-6:]))  # show last 6 lines

        with st.spinner("Processing documents..."):
            # Write uploaded files to a temp directory so pypdf can read them by path
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_paths = []
                for uf in uploaded_files:
                    tmp_path = Path(tmp_dir) / uf.name
                    tmp_path.write_bytes(uf.getbuffer())
                    tmp_paths.append(tmp_path)

                try:
                    num_chunks = ingest_pdfs(
                        tmp_paths,
                        clear_existing=replace_existing,
                        progress_callback=update_status,
                    )
                    st.success(f"Ingested {len(tmp_paths)} file(s), {num_chunks} chunks total.")
                    if replace_existing:
                        st.session_state.ingested_files = [uf.name for uf in uploaded_files]
                    else:
                        st.session_state.ingested_files.extend(uf.name for uf in uploaded_files)
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested_files:
        st.markdown("**Documents in knowledge base (this session):**")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- {fname}")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "DocuMind is an agentic RAG system: it routes queries, retrieves with "
        "hybrid search + reranking, grades its own retrieval, and checks its "
        "own answers for groundedness before responding."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

question = st.text_input(
    "Ask a question:",
    placeholder="e.g. what is a Savonius turbine?",
)
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        try:
            result = compiled_graph.invoke({"question": question})
            st.session_state.history.insert(0, {"question": question, "result": result})
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --- Display results, most recent first ---
for entry in st.session_state.history:
    q = entry["question"]
    result = entry["result"]

    st.markdown("---")
    st.markdown(f"**Q: {q}**")
    st.write(result.get("answer", "(no answer generated)"))

    # --- Transparency panel: shows the agentic behavior actually happened ---
    with st.expander("🔍 See how DocuMind arrived at this answer"):
        route = result.get("route", "unknown")
        retry_count = result.get("retry_count", 0)
        groundedness_pass = result.get("groundedness_pass")
        citations = result.get("citations", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Route taken", route)
        with col2:
            st.metric("Retrieval retries", retry_count)
        with col3:
            gr_label = "✅ Grounded" if groundedness_pass else "⚠️ Not fully grounded"
            st.metric("Groundedness", gr_label)

        if citations:
            st.markdown(f"**Sources used:** {', '.join(citations)}")

        if result.get("sub_questions"):
            st.markdown("**Question was broken down into:**")
            for sq in result["sub_questions"]:
                st.markdown(f"- {sq}")

        retrieved_chunks = result.get("retrieved_chunks", [])
        if retrieved_chunks:
            st.markdown("**Retrieved chunks (after hybrid search + rerank):**")
            for i, chunk in enumerate(retrieved_chunks, 1):
                score = chunk.get("rerank_score", 0)
                st.markdown(f"*Chunk {i} — source: {chunk.get('source', '?')} (relevance: {score:.2f})*")
                st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))

        if not groundedness_pass and result.get("groundedness_note"):
            st.warning(f"Groundedness check flagged: {result['groundedness_note']}")