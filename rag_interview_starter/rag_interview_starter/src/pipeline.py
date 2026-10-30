from src.retrieval import retrieve
from src.generation import build_prompt, generate_answer
from src.citations import format_sources

def answer_question(question: str) -> dict:
    contexts = retrieve(question)
    if not contexts:
        return {
            "question": question,
            "answer": "I could not retrieve supporting documents, so I cannot answer reliably.",
            "sources": []
        }
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt)
    return {
        "question": question,
        "answer": answer,
        "sources": format_sources(contexts)
    }
