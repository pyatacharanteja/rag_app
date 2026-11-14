"""Core RAG logic (no Streamlit imports so it can be tested on its own)."""
import io
import re
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
from pypdf import PdfReader

NOT_FOUND_MSG = "I couldn't find that information in the uploaded documents."


@dataclass
class Chunk:
    text: str
    source: str
    page: int


# ---------- PDF loading & chunking ----------
def extract_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # prefer to cut at a sentence / whitespace boundary
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
            else:
                sp = text.rfind(" ", start, end)
                if sp > start + size // 2:
                    end = sp
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(file_bytes: bytes, filename: str) -> List[Chunk]:
    out = []
    for page_no, text in extract_pages(file_bytes):
        for piece in chunk_text(text):
            out.append(Chunk(piece, filename, page_no))
    return out


# ---------- Embeddings ----------
class Embedder:
    """Lazy wrapper around sentence-transformers (local, free, CPU)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vecs = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


# ---------- Vector store (numpy cosine similarity) ----------
class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.matrix = None  # (n, d) normalized

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def sources(self) -> List[str]:
        return sorted({c.source for c in self.chunks})

    def remove_source(self, name: str):
        keep = [i for i, c in enumerate(self.chunks) if c.source != name]
        self.chunks = [self.chunks[i] for i in keep]
        self.matrix = self.matrix[keep] if (keep and self.matrix is not None) else None

    def search(self, qvec: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.matrix is None or not self.chunks:
            return []
        scores = self.matrix @ qvec
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]


# ---------- LLM ----------
def make_groq_llm(api_key: str, model: str = "openai/gpt-oss-120b") -> Callable:
    from groq import Groq
    client = Groq(api_key=api_key)

    def call(messages: List[dict]) -> str:
        kwargs = dict(model=model, messages=messages, temperature=0.1, max_tokens=2500)
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"  # keeps answers fast, leaves room for output
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    return call


SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer ONLY using the "
    "context excerpts provided. Each excerpt is labelled like [file.pdf, p.3]. "
    "Cite the sources you used inline in that same format. If the context does "
    "not contain the answer, reply with exactly: " + NOT_FOUND_MSG + " "
    "Never use outside knowledge. Be concise and accurate."
)

CONDENSE_PROMPT = (
    "Given the chat history and a follow-up question, rewrite the follow-up as a "
    "single standalone question that includes any needed context from the history. "
    "If it is already standalone, return it unchanged. Output ONLY the question."
)


class RAGEngine:
    def __init__(self, embedder, llm: Callable, top_k: int = 5, min_score: float = 0.05):
        self.embedder = embedder
        self.llm = llm
        self.store = VectorStore()
        self.top_k = top_k
        self.min_score = min_score

    def ingest(self, file_bytes: bytes, filename: str) -> int:
        chunks = build_chunks(file_bytes, filename)
        if not chunks:
            return 0
        vecs = self.embedder.encode([c.text for c in chunks])
        self.store.remove_source(filename)  # re-upload replaces old version
        self.store.add(chunks, vecs)
        return len(chunks)

    def condense(self, question: str, history: List[dict]) -> str:
        if not history:
            return question
        recent = history[-6:]
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        msgs = [
            {"role": "system", "content": CONDENSE_PROMPT},
            {"role": "user", "content": f"Chat history:\n{convo}\n\nFollow-up: {question}"},
        ]
        try:
            out = self.llm(msgs).strip()
            return out or question
        except Exception:
            return question

    def answer(self, question: str, history: List[dict]):
        """Returns (answer_text, sources) where sources = [(Chunk, score)]."""
        if not self.store.chunks:
            return "Please upload and process at least one PDF first.", []
        standalone = self.condense(question, history)
        qvec = self.embedder.encode([standalone])[0]
        hits = [h for h in self.store.search(qvec, self.top_k) if h[1] >= self.min_score]
        if not hits:
            return NOT_FOUND_MSG, []
        seen = {(c.source, c.text) for c, _ in hits}
        extra = []
        for src in {c.source for c, _ in hits}:
            first = next((c for c in self.store.chunks if c.source == src), None)
            if first and (first.source, first.text) not in seen:
                extra.append((first, 0.0))
        hits = hits + extra
        context = "\n\n".join(f"[{c.source}, p.{c.page}]\n{c.text}" for c, _ in hits)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        msgs += history[-6:]
        msgs.append({"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {standalone}"})
        text = self.llm(msgs)
        if NOT_FOUND_MSG.lower() in text.lower():
            return NOT_FOUND_MSG, []
        return text, hits
