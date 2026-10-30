# RAG Interview Starter — end-to-end sample

A beginner-friendly, local RAG demo you can open in VS Code. It demonstrates the main stages:

1. Load source documents
2. Clean text and split it into chunks
3. Create embeddings
4. Store vectors and metadata in ChromaDB
5. Embed a user query and retrieve relevant chunks
6. Rerank candidates (optional Cohere integration)
7. Build a grounded prompt
8. Generate an answer (Ollama locally or OpenAI API)
9. Return source citations
10. Run a tiny evaluation / smoke test

This is a learning project, not production-ready. It uses sample documents so you can run it without uploading confidential data.

## Requirements

- Python 3.11 or 3.12 recommended
- VS Code
- Optional: Ollama installed locally, or an OpenAI API key
- Optional: Cohere API key for reranking

## 1. Open the folder

Extract the ZIP, then in VS Code choose **File → Open Folder** and select `rag_interview_starter`.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Configure environment

Copy `.env.example` to `.env`.

- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2` (downloads the model on first run; no embedding API key required).
- Default vector store: ChromaDB, persisted locally under `data/chroma`.
- Default LLM provider: Ollama. Install/start Ollama and pull a model, for example:

```powershell
ollama pull llama3.2
```

Set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=llama3.2`.

Alternative: use OpenAI by setting `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`.
Optional reranking: set `COHERE_API_KEY` and `USE_COHERE_RERANK=true`. Without it, the demo uses similarity ranking only.

## 4. Add sample documents

The project includes example business and network troubleshooting documents under `data/documents/`. Replace or add `.txt` files for your own experiments. Do not put confidential employer data in this demo.

## 5. Build the index

```powershell
python -m src.ingest
```

This parses `.txt` files, chunks text, creates embeddings and writes to ChromaDB.

## 6. Ask a question

```powershell
python -m src.ask "What can cause an SD-WAN tunnel to go down?"
python -m src.ask "What was the sample healthcare revenue in Q2?"
```

The output includes the generated answer and the source document IDs / filenames.

## 7. Run the API

```powershell
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser to try the FastAPI endpoint.

## 8. Run a basic smoke test

Build the index first, then:

```powershell
python -m src.evaluate
```

This is a tiny smoke test, not a comprehensive RAG evaluation benchmark.

## Architecture

```text
data/documents/*.txt
        |
        v
src/ingest.py
  load -> clean -> chunk -> embed -> ChromaDB
        |
        v
src/retrieval.py
  query embedding -> similarity search -> optional rerank
        |
        v
src/generation.py
  grounded prompt -> LLM
        |
        v
src/citations.py
  source list -> answer + citations
        |
        v
src/ask.py / src/api.py
```

## Key files

- `src/ingest.py`: ingestion pipeline
- `src/chunking.py`: text cleanup and chunking
- `src/embeddings.py`: local Sentence Transformers embeddings
- `src/vector_store.py`: ChromaDB storage/search
- `src/retrieval.py`: retrieval and optional reranking
- `src/generation.py`: prompt and LLM provider integration
- `src/citations.py`: citation formatting
- `src/pipeline.py`: end-to-end orchestration
- `src/api.py`: FastAPI endpoint
- `src/evaluate.py`: tiny query smoke test

## Important notes

- Query and document embeddings must be compatible. This project uses the same embedding model for both.
- Similarity is not a probability; a high score does not guarantee relevance.
- Citations identify retrieved sources but do not guarantee every claim is supported.
- Never expose API keys, never put secrets in source control, and do not send confidential business/network data to external services without authorization.
- Production systems also need authentication, authorization, robust evaluation, monitoring, rate limits, retries, backups and data-retention controls.
