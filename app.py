import hashlib
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

from embeddings.embedding_model import EmbeddingModel
from embeddings.vector_store import VectorStore
from retrieval.bm25_retriever import BM25Retriever
from services.indexing import IndexingService, NoExtractableContentError
from services.qa import QAService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# CSS - clean, light, high-contrast, one accent color
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #FFFFFF;
    --surface: #F7F8FA;
    --border: #E3E6EA;
    --text: #1A1D23;
    --text-muted: #6B7280;
    --accent: #2563EB;
    --accent-soft: #EFF4FF;
    --user-bubble: #2563EB;
    --user-bubble-text: #FFFFFF;
    --assistant-bubble: #F3F4F6;
    --assistant-bubble-text: #1A1D23;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background-color: var(--bg); }

[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }

h1, h2, h3 { color: var(--text); font-weight: 600; }
p, span, label, div { color: var(--text); }

.subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}

.stButton > button {
    font-weight: 500;
    border-radius: 6px;
    border: 1px solid var(--border);
    background-color: #FFFFFF;
    color: var(--text);
}
.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF;
    border: 1.5px dashed var(--border);
    border-radius: 8px;
}

[data-testid="stTabs"] button[role="tab"] {
    font-weight: 500;
    color: var(--text-muted);
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
}

[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background-color: #FFFFFF;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    background-color: #FFFFFF !important;
    caret-color: var(--text) !important;
    unicode-bidi: plaintext;
    text-align: start;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

.msg-row { display: flex; margin: 0.7rem 0; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }

.msg-bubble {
    max-width: 70%;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    line-height: 1.55;
    font-size: 0.98rem;
    unicode-bidi: plaintext;
    text-align: start;
}
.msg-bubble.user {
    background-color: var(--user-bubble);
    border-bottom-right-radius: 3px;
}
.msg-bubble.assistant {
    background-color: var(--assistant-bubble);
    border-bottom-left-radius: 3px;
}
.msg-bubble.user, .msg-bubble.user * { color: var(--user-bubble-text) !important; }
.msg-bubble.assistant, .msg-bubble.assistant * { color: var(--assistant-bubble-text) !important; }

.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.4rem 0 0.2rem 0;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    unicode-bidi: plaintext;
    text-align: start;
}

.source-badge {
    unicode-bidi: plaintext;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.78rem;
    color: var(--accent) !important;
    background-color: var(--accent-soft);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
}

.status-box {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.8rem;
    text-align: center;
    color: var(--text-muted);
    margin-top: 1rem;
    background-color: var(--surface);
}
.status-box strong { color: var(--text); }

.doc-chip {
    display: inline-block;
    font-size: 0.82rem;
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    background-color: var(--accent-soft);
    color: var(--accent) !important;
    margin-bottom: 0.6rem;
}

button:focus-visible, textarea:focus-visible, input:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# Session State
#
# Every uploaded file gets its own independent index, QA service,
# chat history, summary, and suggested questions - nothing is
# pooled or searched across files. "documents" below is keyed by
# filename.
# ============================================================

DEFAULTS = {
    "documents": {},              # filename -> {index_data, qa_service}
    "chat_histories": {},         # filename -> list of messages
    "summaries": {},              # filename -> summary text
    "questions_map": {},          # filename -> suggested questions text
    "processed_signatures": {},   # filename -> (name, size) already indexed
    "deleted_files": set(),       # filenames explicitly removed by the user
    "selected_file": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# A plain module-level scratch directory (not session_state) - it
# needs to be readable from ThreadPoolExecutor worker threads during
# parallel indexing, and st.session_state isn't accessible outside
# the main Streamlit script thread.
TEMP_DIR = tempfile.mkdtemp(prefix="doc_intel_")


CACHE_DIR = Path(__file__).parent / ".index_cache"
CACHE_DIR.mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file):

    dest = Path(TEMP_DIR) / uploaded_file.name

    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dest)


def _content_hash(uploaded_file):
    """Hash file content (not just name) so a renamed-but-identical
    file still hits the cache, and an edited file with the same name
    correctly misses it."""

    uploaded_file.seek(0)
    digest = hashlib.md5(uploaded_file.read()).hexdigest()[:16]
    uploaded_file.seek(0)

    return digest


def build_single_document_index(uploaded_file):
    """Index exactly one file in isolation, so it never gets pooled
    with any other document. Reuses a cached FAISS index from disk
    when this exact file has been indexed before, instead of paying
    for embedding again."""

    digest = _content_hash(uploaded_file)
    index_path = CACHE_DIR / f"{digest}.faiss"
    meta_path = CACHE_DIR / f"{digest}.pkl"

    cached_store = VectorStore.load(str(index_path), str(meta_path))

    if cached_store is not None:

        # The cache key is based on file content only, not which
        # embedding model produced the vectors. If the embedding
        # model has changed since this was cached (different vector
        # dimension), using it would crash FAISS at query time with
        # a cryptic "assert d == self.d". Detect that here and treat
        # it as a cache miss instead.
        expected_dim = (
            EmbeddingModel().model.get_sentence_embedding_dimension()
        )

        if cached_store.index.d != expected_dim:
            logger.warning(
                "Cached index for %s has dimension %d, expected %d "
                "(embedding model changed) - rebuilding.",
                uploaded_file.name, cached_store.index.d, expected_dim
            )
            cached_store = None

    if cached_store is not None:

        # BM25 has no persistence of its own, but rebuilding it from
        # already-chunked text is cheap (just tokenizing) - no
        # embedding or LLM calls involved, so this is fast even for
        # a cache hit.
        bm25_store = BM25Retriever()
        bm25_store.add(cached_store.documents)

        index_data = {
            "vector_store": cached_store,
            "bm25_store": bm25_store,
            "documents": 1,
            "chunks": len(cached_store.documents),
            "failed_files": []
        }

    else:

        file_path = save_uploaded_file(uploaded_file)

        # A fresh IndexingService per file keeps vector stores/BM25
        # indexes fully separate from one another.
        index_data = IndexingService().build_index([file_path])

        index_data["vector_store"].save(str(index_path), str(meta_path))

    qa_service = QAService(
        vector_store=index_data["vector_store"],
        bm25_store=index_data["bm25_store"]
    )

    return index_data, qa_service


def forget_document(filename):

    st.session_state.documents.pop(filename, None)
    st.session_state.chat_histories.pop(filename, None)
    st.session_state.summaries.pop(filename, None)
    st.session_state.questions_map.pop(filename, None)
    st.session_state.processed_signatures.pop(filename, None)


def delete_document(filename):
    """User-initiated removal. Unlike forget_document (used when a
    file disappears from the uploader), this also marks the filename
    as deleted so it isn't silently re-indexed on the next rerun
    while it's still sitting in the file_uploader widget."""

    forget_document(filename)
    st.session_state.deleted_files.add(filename)

    if st.session_state.selected_file == filename:
        st.session_state.selected_file = None


def render_sources(sources):

    if not sources:
        return

    # Multiple retrieved chunks often land on the same page (e.g. a
    # short document chunked into several pieces) - one badge per
    # unique (file, page) is enough, not one per chunk.
    seen = set()
    unique_sources = []

    for s in sources:

        key = (s["file"], s["page"])

        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    badges = "".join(
        f'<span class="source-badge">📄 {s["file"]} · p.{s["page"]}</span>'
        for s in unique_sources
    )

    st.markdown(f'<div class="source-row">{badges}</div>', unsafe_allow_html=True)


NO_ANSWER_MESSAGES = {
    "I couldn't find the answer in the uploaded documents.",
    "لم أتمكن من العثور على الإجابة في المستندات المرفوعة."
}


def render_message(role, content, sources=None):

    st.markdown(
        f'<div class="msg-row {role}">'
        f'<div class="msg-bubble {role}">{content}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Don't show citations alongside a "couldn't find an answer" reply -
    # the retriever always returns its best-scoring chunks even when
    # nothing is actually relevant (e.g. "hi", "how are you"), so
    # showing them here would contradict what the assistant just said.
    if role == "assistant" and sources and content.strip() not in NO_ANSWER_MESSAGES:
        render_sources(sources)


# ============================================================
# Sidebar - Upload (each file indexed independently)
# ============================================================

with st.sidebar:

    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "pptx"],
        accept_multiple_files=True
    )

    if st.button("Clear all", use_container_width=True):

        for key, default in DEFAULTS.items():
            st.session_state[key] = default.copy() if isinstance(default, (dict, set)) else default

        st.rerun()

    current_by_name = {f.name: f for f in uploaded_files} if uploaded_files else {}

    # Drop documents that were removed from the uploader widget, and
    # forget the "user deleted this" marker too, so re-adding the
    # same-named file later is treated as fresh.
    for existing_name in list(st.session_state.documents.keys()):
        if existing_name not in current_by_name:
            forget_document(existing_name)

    for deleted_name in list(st.session_state.deleted_files):
        if deleted_name not in current_by_name:
            st.session_state.deleted_files.discard(deleted_name)

    # Index any file that is new or changed, skipping anything the
    # user explicitly deleted (it may still be sitting in the
    # uploader widget even though we don't want to re-index it).
    # Each file is fully independent (separate vector store, separate
    # BM25 index), so they're built concurrently.
    to_process = [
        (name, f) for name, f in current_by_name.items()
        if name not in st.session_state.deleted_files
        and st.session_state.processed_signatures.get(name) != (name, f.size)
    ]

    if to_process:

        with st.spinner(
            f"Reading {len(to_process)} document(s)..."
        ):

            results = {}

            with ThreadPoolExecutor(max_workers=min(4, len(to_process))) as pool:

                futures = {
                    pool.submit(build_single_document_index, uploaded_file): name
                    for name, uploaded_file in to_process
                }

                for future in as_completed(futures):

                    name = futures[future]

                    try:
                        results[name] = ("ok", future.result())

                    except NoExtractableContentError as exc:
                        results[name] = ("error", str(exc))

                    except Exception as exc:
                        logger.exception("Index build failed for %s", name)
                        results[name] = ("error", str(exc))

            # Apply results on the main thread, in upload order, so
            # session_state writes stay predictable.
            for name, uploaded_file in to_process:

                status, payload = results[name]

                if status == "error":
                    st.error(f"{name}: {payload}")
                    continue

                index_data, qa_service = payload

                st.session_state.documents[name] = {
                    "index_data": index_data,
                    "qa_service": qa_service
                }

                st.session_state.chat_histories[name] = []
                st.session_state.summaries[name] = None
                st.session_state.questions_map[name] = None
                st.session_state.processed_signatures[name] = (
                    name, uploaded_file.size
                )

                if st.session_state.selected_file is None:
                    st.session_state.selected_file = name

    if st.session_state.documents:

        st.markdown("<hr>", unsafe_allow_html=True)
        st.success(f"{len(st.session_state.documents)} document(s) ready")

        for name, doc in list(st.session_state.documents.items()):

            row_col, delete_col = st.columns([5, 1])

            with row_col:
                st.caption(name)

            with delete_col:
                if st.button("🗑️", key=f"delete_{name}", help=f"Remove {name}"):
                    delete_document(name)
                    st.rerun()

            failed = doc["index_data"].get("failed_files", [])

            if failed:
                st.caption(f"⚠️ {name} had extraction issues")


# ============================================================
# Main Area
# ============================================================

st.title("Document Assistant")
st.markdown(
    '<div class="subtitle">Upload documents in the sidebar, pick one below, '
    'and ask questions, get a summary, or browse suggested questions - '
    'scoped to that document only.</div>',
    unsafe_allow_html=True
)

if not st.session_state.documents:

    st.markdown(
        '<div class="status-box"><strong>No documents yet</strong><br>'
        'Upload a file in the sidebar to get started.</div>',
        unsafe_allow_html=True
    )

else:

    doc_names = list(st.session_state.documents.keys())

    if st.session_state.selected_file not in doc_names:
        st.session_state.selected_file = doc_names[0]

    selected = st.selectbox(
        "Chatting with",
        doc_names,
        index=doc_names.index(st.session_state.selected_file)
    )
    st.session_state.selected_file = selected

    st.markdown(f'<span class="doc-chip">📄 {selected}</span>', unsafe_allow_html=True)

    doc = st.session_state.documents[selected]
    qa_service = doc["qa_service"]
    index_data = doc["index_data"]

    tab_chat, tab_summary, tab_questions = st.tabs(
        ["Chat", "Summary", "Suggested Questions"]
    )

    # --------------------------------------------------------
    # Chat Tab
    # --------------------------------------------------------

    with tab_chat:

        history = st.session_state.chat_histories[selected]

        for message in history:

            render_message(
                message["role"],
                message["content"],
                message.get("sources")
            )

        question = st.chat_input(f"Ask a question about {selected}...")

        if question:

            history.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):

                try:
                    result = qa_service.answer(question)

                except Exception as exc:
                    logger.exception("QA failed")
                    result = {
                        "answer": f"Something went wrong: {exc}",
                        "sources": []
                    }

            history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                }
            )

            st.rerun()

    # --------------------------------------------------------
    # Summary Tab
    # --------------------------------------------------------

    with tab_summary:

        if st.button("Generate summary", key=f"summary_btn_{selected}"):

            with st.spinner(f"Summarizing {selected}..."):

                try:

                    summary = IndexingService().summarize_documents(
                        index_data["vector_store"].documents
                    )

                    st.session_state.summaries[selected] = summary

                except Exception as exc:
                    logger.exception("Summary generation failed")
                    st.error(f"Couldn't generate a summary: {exc}")

        if st.session_state.summaries.get(selected):
            st.markdown(st.session_state.summaries[selected])

    # --------------------------------------------------------
    # Suggested Questions Tab
    # --------------------------------------------------------

    with tab_questions:

        if st.button("Generate suggested questions", key=f"questions_btn_{selected}"):

            with st.spinner(f"Coming up with questions about {selected}..."):

                try:

                    questions = IndexingService().generate_questions(
                        index_data["vector_store"].documents
                    )

                    st.session_state.questions_map[selected] = questions

                except Exception as exc:
                    logger.exception("Question generation failed")
                    st.error(f"Couldn't generate questions: {exc}")

        if st.session_state.questions_map.get(selected):

            st.markdown(st.session_state.questions_map[selected])

            st.caption(
                "Copy any question above into the Chat tab to get an answer."
            )