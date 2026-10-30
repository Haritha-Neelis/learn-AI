from src.config import TOP_K, RERANK_TOP_N, USE_COHERE_RERANK, COHERE_API_KEY, COHERE_RERANK_MODEL
from src.vector_store import search

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    \"\"\"Optional Cohere reranking. If disabled, keep vector-search order.\"\"\"
    if not USE_COHERE_RERANK:
        return candidates[:RERANK_TOP_N]
    if not COHERE_API_KEY:
        raise RuntimeError("USE_COHERE_RERANK=true but COHERE_API_KEY is empty")
    import cohere
    client = cohere.ClientV2(api_key=COHERE_API_KEY)
    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[item["text"] for item in candidates],
        top_n=min(RERANK_TOP_N, len(candidates))
    )
    ranked = []
    for result in response.results:
        item = dict(candidates[result.index])
        item["rerank_score"] = float(result.relevance_score)
        ranked.append(item)
    return ranked

def retrieve(query: str) -> list[dict]:
    candidates = search(query, top_k=TOP_K)
    return rerank(query, candidates)
