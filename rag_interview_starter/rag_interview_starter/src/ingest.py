from pathlib import Path
from src.config import DOCUMENTS_PATH
from src.chunking import clean_text, chunk_text
from src.vector_store import add_chunks

def load_text_documents() -> list[dict]:
    documents = []
    for path in sorted(DOCUMENTS_PATH.glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8"))
        if text:
            documents.append({"source": path.name, "text": text})
    return documents

def build_chunks() -> list[dict]:
    chunks = []
    for doc in load_text_documents():
        for index, piece in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "id": f'{doc["source"]}::chunk-{index}',
                "text": piece,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": index,
                    "document_type": "sample"
                }
            })
    return chunks

def main():
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"No .txt documents found in {DOCUMENTS_PATH}")
    count = add_chunks(chunks)
    print(f"Indexed {count} chunks into ChromaDB.")
    print("Sources:", ", ".join(sorted({c["metadata"]["source"] for c in chunks})))

if __name__ == "__main__":
    main()
