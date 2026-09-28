from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm import call_llm
from app.retrieval import Retriever, load_chunks

app = FastAPI(title="My Assistant")


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    answer: str
    sources: list[str]


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(load_chunks())


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    hits = get_retriever().search(q.text)
    if not hits:
        return Answer(answer="مش لاقي إجابة في الملاحظات.", sources=[])
    answer = call_llm(build_prompt(q.text, hits))
    return Answer(answer=answer, sources=sorted({h["source"] for h in hits}))
