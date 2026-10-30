import chromadb
from src.config import CHROMA_PATH, COLLECTION_NAME
from src.embeddings import embed_documents, embed_query

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    vectors = embed_documents([item["text"] for item in chunks])
    collection.upsert(
        ids=[item["id"] for item in chunks],
        documents=[item["text"] for item in chunks],
        embeddings=vectors,
        metadatas=[item["metadata"] for item in chunks]
    )
    return len(chunks)

def search(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    query_vector = embed_query(query)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    found = []
    for i, doc in enumerate(result["documents"][0]):
        found.append({
            "text": doc,
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i]
        })
    return found
