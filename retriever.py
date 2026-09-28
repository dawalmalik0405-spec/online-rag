from vectordb import get_document_collection, embeddings


def search_query(query: str, collection=None, embeddings_model=None, filename: str = None, top_k: int = 6):
    """
    Performs semantic retrieval against the isolated Chroma collection for the given filename.
    Dynamically bounds n_results to avoid ChromaDB count overflow errors.
    """
    if not filename:
        raise ValueError("Filename is required for retrieval")

    embed_model = embeddings_model or embeddings
    col = collection if collection is not None else get_document_collection(filename)

    total_chunks = col.count()
    if total_chunks == 0:
        return []

    # Dynamically cap n_results to avoid Chroma count errors
    n_results = min(top_k, total_chunks)

    query_embedding = embed_model.embed_query(query)

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }

    results = col.query(**query_args)

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "content": doc,
            "metadata": meta,
            "similarity": round(max(0.0, 1.0 - dist), 4)
        })

    return chunks
