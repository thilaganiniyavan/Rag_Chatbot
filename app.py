"""
RAG Application - Streamlit Frontend
"""

import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="RAG System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .answer-box {
        background: #1e1e2e;
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        color: #e2e8f0;
        line-height: 1.7;
    }
    .source-chip {
        display: inline-block;
        background: #134e4a;
        border: 1px solid #0d9488;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.78rem;
        margin: 0.15rem;
        color: #99f6e4;
    }
    .user-bubble {
        background: #1e1b4b;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: #e2e8f0;
    }
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


def get_kb_count():
    try:
        from rag_app.vector_store import get_collection_info
        return get_collection_info()["count"]
    except Exception:
        return 0


with st.sidebar:
    st.title("RAG System")
    st.divider()

    count = get_kb_count()
    if count > 0:
        st.success(f"{count} chunks indexed")
    else:
        st.info("Knowledge base is empty")

    st.divider()
    st.subheader("Upload Documents")
    st.caption("Supported: PDF, DOCX, TXT, CSV, JSON, HTML, MD, XLSX")

    uploaded = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "txt", "csv", "json", "html", "htm", "md", "markdown", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚡ Process & Index", use_container_width=True, type="primary"):
            from rag_app.document_processor import load_and_chunk_bytes
            from rag_app.vector_store import add_documents

            all_chunks = []
            progress = st.progress(0)
            status = st.empty()

            for i, f in enumerate(uploaded):
                status.text(f"Processing: {f.name}…")
                try:
                    chunks = load_and_chunk_bytes(f.read(), f.name)
                    all_chunks.extend(chunks)
                    if f.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(f.name)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                progress.progress((i + 1) / len(uploaded))

            if all_chunks:
                status.text("Embedding and storing…")
                try:
                    add_documents(all_chunks)
                    status.empty()
                    progress.empty()
                    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded)} file(s)!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Storage error: {e}")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Indexed Files")
        for f in st.session_state.uploaded_files:
            st.markdown(f"- `{f}`")

    st.divider()
    if st.button("Clear Knowledge Base", use_container_width=True):
        from rag_app.vector_store import clear_vectorstore
        clear_vectorstore()
        st.session_state.uploaded_files = []
        st.session_state.chat_history = []
        st.success("Cleared.")
        st.rerun()

    st.caption("Answers are strictly from your documents only.")


st.title("RAG System")
st.markdown(
    "Ask questions about your uploaded documents. "
    "Answers are **strictly grounded** in your knowledge base — no hallucination."
)
st.divider()

# Chat history
if st.session_state.chat_history:
    for entry in st.session_state.chat_history:
        st.markdown(f'<div class="user-bubble"><strong>You:</strong> {entry["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><strong>Assistant:</strong><br>{entry["answer"]}</div>', unsafe_allow_html=True)

        if entry.get("sources"):
            chips = "".join(
                f'<span class="source-chip">{s["filename"]}'
                + (f' · p{s["page"]}' if s.get("page") else "")
                + "</span>"
                for s in entry["sources"]
            )
            st.markdown(f"<small>Sources: {chips}</small>", unsafe_allow_html=True)

            with st.expander("View source snippets"):
                for j, src in enumerate(entry["sources"], 1):
                    st.markdown(f"**{j}. {src['filename']}" + (f" (page {src['page']})" if src.get("page") else "") + "**")
                    st.caption(src["snippet"])
                    if j < len(entry["sources"]):
                        st.divider()

    st.divider()

# Query input
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the types of machine learning?",
        label_visibility="collapsed",
    )
with col2:
    ask_btn = st.button("Ask", use_container_width=True, type="primary")

# Handle query
if ask_btn and question:
    kb_count = get_kb_count()
    if kb_count == 0:
        st.warning("Upload and index documents first using the sidebar.")
    else:
        with st.spinner("Retrieving context and generating answer…"):
            try:
                from rag_app.rag_chain import ask
                result = ask(question)
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
            except ValueError as e:
                st.error(f"Config error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
elif ask_btn and not question:
    st.warning("Please enter a question.")

# Empty state
if not st.session_state.chat_history:
    kb_count = get_kb_count()
    if kb_count == 0:
        st.markdown('<div class="empty-state">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("** Step 1**\n\nUpload documents using the sidebar (PDF, DOCX, TXT, CSV, JSON, HTML, MD)")
        with col2:
            st.info("** Step 2**\n\nClick **Process & Index** to embed and store them in ChromaDB")
        with col3:
            st.info("** Step 3**\n\nAsk questions above — answers come only from your documents")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success(f"Knowledge base ready with {kb_count} chunks. Ask a question above!")
