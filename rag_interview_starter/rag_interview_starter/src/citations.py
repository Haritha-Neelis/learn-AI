def format_sources(contexts: list[dict]) -> list[dict]:
    \"\"\"Return source metadata for UI/API display. This doesn't prove claim support.\"\"\"
    sources = []
    seen = set()
    for item in contexts:
        metadata = item.get("metadata", {})
        key = (metadata.get("source", "unknown"), metadata.get("chunk_index", -1))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", -1)
        })
    return sources
