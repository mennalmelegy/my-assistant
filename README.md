# My Assistant

Small RAG-style assistant built with FastAPI. Retrieves relevant chunks from local notes using TF-IDF, then answers via Groq LLM, with a refusal path when nothing relevant is found.

## Run locally

pip install -r requirements.txt
$env:GROQ_API_KEY = "your_key"
uvicorn app.main:app --reload

Open http://127.0.0.1:8000/docs

## Test

pytest -v
