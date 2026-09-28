from fastapi.testclient import TestClient

from app import main
from app.retrieval import Retriever

client = TestClient(main.app)

CHUNKS = [
    {"source": "ml.md", "text": "Overfitting happens when a model memorizes training data and fails on new data."},
    {"source": "db.md", "text": "An index speeds up SELECT queries on large tables in Postgres."},
]


def fake_setup(monkeypatch, llm=lambda p: "fake answer"):
    monkeypatch.setattr(main, "get_retriever", lambda: Retriever(CHUNKS))
    monkeypatch.setattr(main, "call_llm", llm)


def test_empty_question_returns_400(monkeypatch):
    fake_setup(monkeypatch)
    assert client.post("/ask", json={"text": "   "}).status_code == 400


def test_missing_field_returns_422(monkeypatch):
    fake_setup(monkeypatch)
    assert client.post("/ask", json={}).status_code == 422


def test_answer_includes_sources(monkeypatch):
    fake_setup(monkeypatch)
    r = client.post("/ask", json={"text": "what is overfitting?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "fake answer", "sources": ["ml.md"]}


def test_no_match_refuses_without_calling_llm(monkeypatch):
    def boom(prompt):
        raise AssertionError("LLM should not be called")

    fake_setup(monkeypatch, llm=boom)
    r = client.post("/ask", json={"text": "quantum banana"})
    assert r.status_code == 200
    assert r.json()["sources"] == []
