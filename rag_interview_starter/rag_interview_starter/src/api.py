from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.pipeline import answer_question

app = FastAPI(title="RAG Interview Starter", version="0.1.0")

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(request: AskRequest):
    try:
        return answer_question(request.question)
    except Exception as exc:
        # Demo only: production should log detailed errors privately and return
        # sanitized error messages to clients.
        raise HTTPException(status_code=503, detail="RAG request failed. Check server logs.") from exc
