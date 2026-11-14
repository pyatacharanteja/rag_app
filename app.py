import os

import streamlit as st

from rag import Embedder, RAGEngine, make_groq_llm

st.set_page_config(page_title="Multi-Document RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder():
    return Embedder()


def get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Settings")
api_key = get_api_key() or st.sidebar.text_input(
    "Groq API key", type="password", help="Free key from console.groq.com"
)
model = st.sidebar.text_input("LLM model", value="openai/gpt-oss-120b")
top_k = st.sidebar.slider("Chunks retrieved (top-k)", 2, 10, 5)
min_score = st.sidebar.slider("Relevance threshold", 0.0, 0.6, 0.05, 0.01)

if not api_key:
    st.title("📚 Multi-Document RAG Assistant")
    st.info("Enter your Groq API key in the sidebar to begin.")
    st.stop()

# ---------------- Session state ----------------
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine(get_embedder(), llm=None)
    st.session_state.messages = []
    st.session_state.processed = {}  # name -> (size, n_chunks)

engine: RAGEngine = st.session_state.engine
try:
    engine.llm = make_groq_llm(api_key, model)
except Exception as e:
    st.error(f"Could not initialise the LLM client: {e}")
    st.stop()
engine.top_k, engine.min_score = top_k, min_score

# ---------------- Upload ----------------
st.title("📚 Multi-Document RAG Assistant")
st.caption("Upload PDFs, then ask questions. Answers are grounded in your documents with page-level sources.")

files = st.sidebar.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Process documents", type="primary", disabled=not files):
    for f in files:
        data = f.getvalue()
        if st.session_state.processed.get(f.name, (None,))[0] == len(data):
            continue
        with st.spinner(f"Indexing {f.name}..."):
            try:
                n = engine.ingest(data, f.name)
            except Exception as e:
                st.sidebar.error(f"{f.name}: failed to read ({e})")
                continue
        if n == 0:
            st.sidebar.warning(f"{f.name}: no extractable text (scanned PDF?).")
        else:
            st.session_state.processed[f.name] = (len(data), n)
            st.sidebar.success(f"{f.name}: {n} chunks indexed")

if st.session_state.processed:
    st.sidebar.markdown("**Indexed documents**")
    for name, (_, n) in st.session_state.processed.items():
        st.sidebar.write(f"• {name} ({n} chunks)")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------- Chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for src in m.get("sources", []):
            with st.expander(f"📄 {src['source']} — page {src['page']} (score {src['score']:.2f})"):
                st.write(src["text"])

question = st.chat_input("Ask a question about your documents...")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                answer, hits = engine.answer(question, history)
            except Exception as e:
                answer, hits = f"Something went wrong calling the LLM: {e}", []
        st.markdown(answer)
        srcs = [{"source": c.source, "page": c.page, "score": s, "text": c.text} for c, s in hits]
        for src in srcs:
            with st.expander(f"📄 {src['source']} — page {src['page']} (score {src['score']:.2f})"):
                st.write(src["text"])
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": srcs})
