# Studify Agent Backend

A backend-only rebuild of the original Studify prototype. This version drops the Android client and turns the project into a cleaner portfolio-ready API built with `FastAPI`, `LangChain`, `ChromaDB`, and PDF ingestion utilities.

## Why this version is stronger

- Backend only: no Kotlin or XML required
- Clean API surface for ingestion, Q&A, summarization, and quiz generation
- Secrets moved to environment variables
- LangChain-powered retrieval and prompt pipelines
- ChromaDB-backed semantic search for course notes
- Easier to demo, test, and extend into LangGraph later

## Features

- Ingest subject PDFs into a Chroma vector store
- Ask subject-specific questions using retrieval-augmented generation
- Summarize uploaded PDFs
- Generate quiz questions from uploaded PDFs
- Keep compatibility aliases for the original Android-facing endpoints

## Project structure

```text
studify-agent-backend/
├── app/
│   ├── api/routes.py
│   ├── core/config.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── services/llm.py
│   ├── services/pdf_utils.py
│   ├── services/study_service.py
│   └── services/vector_store.py
├── tests/test_text_utils.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` if you want to use OpenAI. If you prefer Hugging Face inference, set:

- `LLM_PROVIDER=huggingface`
- `HUGGINGFACE_API_KEY`

## Run

```bash
uvicorn app.main:app --reload --app-dir .
```

## Example flow

### 1. Ingest notes

```bash
curl -X POST "http://127.0.0.1:8000/api/notes/ingest" \
  -F "subject=DBMS" \
  -F "pdf_file=@./data/DBMS_M1.pdf"
```

### 2. Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/api/agent/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is normalization?","subject":"DBMS"}'
```

### 3. Generate a summary

```bash
curl -X POST "http://127.0.0.1:8000/api/content/summary" \
  -F "pdf_file=@./data/DBMS_M1.pdf"
```

### 4. Generate quiz questions

```bash
curl -X POST "http://127.0.0.1:8000/api/content/questions" \
  -F "pdf_file=@./data/DBMS_M1.pdf" \
  -F "question_count=5"
```

## API endpoints

- `GET /health`
- `POST /api/notes/ingest`
- `POST /api/agent/ask`
- `POST /api/content/summary`
- `POST /api/content/questions`

Compatibility aliases:

- `POST /ask_question`
- `POST /generate_summary`
- `POST /generate_questions`

## Notes for interviews

This project demonstrates:

- Python backend development
- REST API design
- RAG fundamentals with ChromaDB
- LangChain prompt and retrieval pipelines
- File upload handling
- LLM application design with configurable providers

The next logical upgrade is replacing the linear service flow with `LangGraph` for explicit state transitions such as retrieve, validate context, answer, and fallback.

