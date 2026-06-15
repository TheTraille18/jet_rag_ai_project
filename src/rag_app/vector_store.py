from pathlib import Path

import chromadb


def get_collection(db_path: str | Path, collection_name: str):
    """Create or load a Chroma collection configured for cosine search."""
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    collection,
    chunks: list[str],
    embeddings: list[list[float]],
    id_prefix: str,
) -> None:
    """Store text chunks in Chroma."""
    for index, chunk in enumerate(chunks, start=1):
        collection.upsert(
            ids=[f"{id_prefix}_{index}"],
            documents=[chunk],
            embeddings=[embeddings[index - 1]],
        )


def query_collection(collection, query_embedding: list[float], n_results: int = 3):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )
