"""Offline unit tests. A hashed bag-of-words embedder and a fake LLM stand in for
MiniLM and Groq, so the pipeline logic is tested without network access.
Run:  pytest -q   (needs: pip install pytest)
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import NOT_FOUND_MSG, RAGEngine, chunk_text, extract_pages  # noqa: E402

SAMPLES = ROOT / "sample_documents"


class FakeEmbedder:
    def encode(self, texts):
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1
        return out / n


def fake_llm(messages):
    if messages[0]["content"].startswith("Given the chat history"):
        return "What was the revenue of Northwind Energy in fiscal year 2025?"
    ctx, _, question = messages[-1]["content"].lower().partition("question:")
    if "revenue" in question and "8,450" in ctx:
        return "Revenue was INR 8,450 crore [northwind_energy_annual_report_2025.pdf, p.1]."
    return NOT_FOUND_MSG


def make_engine():
    e = RAGEngine(FakeEmbedder(), fake_llm, min_score=0.05)
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        assert e.ingest(pdf.read_bytes(), pdf.name) > 0
    return e


def test_extract_pages_keeps_page_numbers():
    data = (SAMPLES / "northwind_energy_annual_report_2025.pdf").read_bytes()
    assert [p for p, _ in extract_pages(data)] == [1, 2, 3]


def test_chunking_respects_size_and_overlap():
    chunks = chunk_text("Sentence number one. " * 200, size=900, overlap=150)
    assert len(chunks) > 1 and all(len(c) <= 900 for c in chunks)


def test_multi_document_ingestion():
    assert len(make_engine().store.sources()) == 3


def test_answer_has_source_reference():
    answer, hits = make_engine().answer("What was Northwind revenue in 2025?", [])
    assert "8,450" in answer
    assert any(c.source == "northwind_energy_annual_report_2025.pdf" for c, _ in hits)


def test_followup_uses_history():
    e = make_engine()
    hist = [{"role": "user", "content": "What was Northwind revenue in 2025?"},
            {"role": "assistant", "content": "INR 8,450 crore."}]
    answer, _ = e.answer("and how does that compare to before?", hist)
    assert "8,450" in answer


def test_unanswerable_returns_not_found():
    answer, hits = make_engine().answer("Who won the football world cup?", [])
    assert answer == NOT_FOUND_MSG and hits == []


def test_no_documents_message():
    answer, _ = RAGEngine(FakeEmbedder(), fake_llm).answer("hello", [])
    assert answer.startswith("Please upload")


def test_reupload_replaces_not_duplicates():
    e = make_engine()
    before = len(e.store.chunks)
    pdf = SAMPLES / "fabrikam_plant_operations_manual.pdf"
    e.ingest(pdf.read_bytes(), pdf.name)
    assert len(e.store.chunks) == before
