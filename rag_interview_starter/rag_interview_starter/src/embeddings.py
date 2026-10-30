from functools import lru_cache
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    # Downloads model weights on first use; no API key required.
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedding_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()

def embed_query(text: str) -> list[float]:
    return embed_documents([text])[0]
