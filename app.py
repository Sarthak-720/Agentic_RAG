"""
app.py
------
Streamlit front-end for CRAG-Ops.

Two things happen here beyond "a chatbot":

1. Dynamic ingestion: the sidebar file uploader lets a user drop in a new
   10-K PDF at any time. It's saved to disk and passed straight through
   `ingest.ingest_file()`, which embeds it into Pinecone (or skips it if
   it's already there) -- no restart, no re-indexing the whole corpus.

2. Transparent self-correction: instead of a single black-box `invoke()`
   call, we stream the LangGraph execution node-by-node (`graph.stream`)
   and surface each step (retrieving, grading, rewriting, web-searching,
   generating, hallucination-checking) in a live status panel, so the
   "agentic" and "self-correcting" behavior is actually visible to the user.
"""

import os
import tempfile

import streamlit as st

from agent import build_graph
from config import get_or_create_index
from ingest import ingest_file, list_ingested_files, reset_knowledge_base

st.set_page_config(page_title="CRAG-Ops | Agentic Financial Analyst", page_icon="💹", layout="wide")

# --------------------------------------------------------------------------
# Cached resources -- built once per app session, not on every rerun
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


@st.cache_resource(show_spinner=False)
def get_index():
    return get_or_create_index()


# Human-friendly labels for each graph node, shown as the agent works.
STEP_LABELS = {
    "retrieve": "🔍 Retrieving relevant chunks from Pinecone...",
    "grade_documents": "🧐 Grading document relevance...",
    "rewrite_query": "✍️ Rewriting the query for web search...",
    "web_search": "🌐 Falling back to a live web search (Tavily)...",
    "generate": "🤖 Generating an answer from the context...",
    "increment_retry": "♻️ Answer wasn't fully grounded -- retrying generation...",
    "fallback": "⚠️ Could not produce a grounded answer, returning a safe response...",
}


def run_agent_streaming(question: str, status_box) -> dict:
    """
    Run the compiled graph via `.stream()` so we can narrate each node as
    it executes, and manually accumulate the state along the way.

    Every node in this graph returns simple key-overwrite updates (no
    LangGraph `Annotated` add-reducers are used in the schema), so a plain
    dict.update() faithfully mirrors LangGraph's own state-merging logic.
    """
    graph = get_graph()
    state = {
        "original_question": question,
        "question": question,
        "generation": "",
        "documents": [],
        "web_search_needed": "No",
        "generation_retries": 0,
    }

    for step in graph.stream(state, stream_mode="updates"):
        for node_name, update in step.items():
            state.update(update)
            status_box.write(STEP_LABELS.get(node_name, f"Running `{node_name}`..."))

    return state


# --------------------------------------------------------------------------
# Sidebar: dynamic knowledge-base management
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.caption("Vectors live in Pinecone (cloud). Nothing is stored locally.")

    uploaded_files = st.file_uploader(
        "Upload a new 10-K PDF",
        type="pdf",
        accept_multiple_files=True,
        help="New filings are embedded into Pinecone immediately and become "
        "queryable on your very next question -- no restart needed. "
        "Note: if a PDF needs the OCR fallback (rare -- only for true scanned "
        "images), ingestion can take several minutes for long documents.",
    )

    if uploaded_files and st.button("Ingest uploaded file(s)", use_container_width=True):
        index = get_index()
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Persist to a temp path -- PyPDFLoader needs a filesystem path.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    result = ingest_file(tmp_path, index=index)
                finally:
                    os.unlink(tmp_path)

            if result["status"] == "ingested":
                st.success(f"✅ {uploaded_file.name}: {result['reason']}")
            else:
                st.info(f"⏭️ {uploaded_file.name}: {result['reason']}")

        # Clear cached document listing so the sidebar reflects new uploads.
        st.cache_data.clear()

    st.divider()
    st.subheader("Currently indexed filings")

    @st.cache_data(ttl=30, show_spinner=False)
    def _cached_file_list():
        return list_ingested_files(get_index())

    indexed_files = _cached_file_list()
    if indexed_files:
        for fname in indexed_files:
            st.markdown(f"- `{fname}`")
    else:
        st.caption("No documents indexed yet. Upload a 10-K PDF above, or run `python ingest.py` "
                   "against a populated `/data` folder.")

    st.divider()
    with st.expander("⚠️ Danger zone: reset knowledge base"):
        st.caption(
            "Permanently deletes every vector currently in Pinecone -- use this "
            "before switching to a completely different set of PDFs. This cannot "
            "be undone."
        )
        confirm_text = st.text_input(
            "Type RESET to confirm", key="reset_confirm", placeholder="RESET"
        )
        if st.button("Delete all indexed documents", type="secondary", use_container_width=True):
            if confirm_text.strip() == "RESET":
                with st.spinner("Wiping the knowledge base..."):
                    result = reset_knowledge_base(get_index())
                if result["status"] == "reset":
                    st.success(result["reason"])
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(result["reason"])
            else:
                st.warning("Type RESET (all caps) in the box above to confirm.")

# --------------------------------------------------------------------------
# Main chat interface
# --------------------------------------------------------------------------
st.title("💹 CRAG-Ops")
st.caption(
    "A self-correcting agentic RAG system for SEC 10-K analysis. "
    "It grades its own retrievals, rewrites queries, falls back to live web "
    "search when the knowledge base can't answer, and checks its own answers "
    "for hallucination before responding."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

question = st.chat_input("Ask about revenue, risk factors, R&D spend, etc...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking...", expanded=True)
        try:
            final_state = run_agent_streaming(question, status_box)
            status_box.update(label="Done", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the user
            status_box.update(label="Something went wrong", state="error")
            st.error(
                f"The agent hit an error: {exc}\n\n"
                "Double-check that OPENAI_API_KEY, PINECONE_API_KEY, and "
                "TAVILY_API_KEY are all set correctly."
            )
            st.stop()

        answer = final_state.get("generation", "I wasn't able to generate an answer.")
        st.markdown(answer)

        sources = sorted({d.metadata.get("source", "unknown") for d in final_state.get("documents", [])})
        if sources:
            with st.expander("Sources used"):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
