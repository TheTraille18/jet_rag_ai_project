from rag_app.bedrock_client import ask_bedrock
from rag_app.config import (
    AWS_REGION,
    COLLECTION_NAME,
    DB_PATH,
    DEFAULT_QUERY,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    MODEL_ID,
    PDF_PATH,
)
from rag_app.pdf_loader import load_pdf
from rag_app.text_splitter import split_text
from rag_app.titan_embeddings import embed_text, embed_texts, get_bedrock_runtime
from rag_app.vector_store import get_collection, ingest_chunks, query_collection


def ensure_pdf_is_indexed(collection, bedrock_runtime) -> None:
    if collection.count() > 0:
        print(f"Using existing Chroma collection with {collection.count()} chunks.")
        return

    print("Collection is empty. Loading PDF and creating Titan embeddings...")

    pdf_text = load_pdf(PDF_PATH)
    chunks = split_text(pdf_text)

    print(f"Loaded PDF with {len(pdf_text)} characters")
    print(f"Split into {len(chunks)} chunks")

    embeddings = embed_texts(
        texts=chunks,
        bedrock_runtime=bedrock_runtime,
        model_id=EMBEDDING_MODEL_ID,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    ingest_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        id_prefix="tesla_earning_chunk",
    )

    print(f"Inserted {len(chunks)} chunks into Chroma.")


def print_match_scores(results) -> None:
    for index, distance in enumerate(results["distances"][0], start=1):
        similarity = 1 - distance
        print(f"Match {index} similarity: {similarity:.4f}")


def main() -> None:
    bedrock_runtime = get_bedrock_runtime(AWS_REGION)

    collection = get_collection(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    ensure_pdf_is_indexed(collection, bedrock_runtime)

    query = DEFAULT_QUERY
    query_embedding = embed_text(
        text=query,
        bedrock_runtime=bedrock_runtime,
        model_id=EMBEDDING_MODEL_ID,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    results = query_collection(collection, query_embedding, n_results=3)

    print_match_scores(results)

    reply = ask_bedrock(
        question=query,
        results=results,
        model_id=MODEL_ID,
        region=AWS_REGION,
    )

    print("Question:")
    print(query)
    print("\nReply:")
    print(reply)
