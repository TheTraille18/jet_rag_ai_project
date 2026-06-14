from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


def get_collection(db_path: str | Path, collection_name: str):
    """Create or load a Chroma collection configured for cosine search."""
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(collection, chunks: list[str], id_prefix: str) -> None:
    """Store text chunks in Chroma."""
    for index, chunk in enumerate(chunks, start=1):
        collection.upsert(
            ids=[f"{id_prefix}_{index}"],
            documents=[chunk],
        )


def query_collection(collection, question: str, n_results: int = 3):
    return collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "distances"],
    )
