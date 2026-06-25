from langchain_aws import ChatBedrockConverse
from rag_app.config import (
    AWS_REGION,
    COLLECTION_NAME,
    DB_PATH,
    DEFAULT_QUERY,
    EMBEDDING_MODEL_ID,
    MODEL_ID,
    PDF_PATH,
)
from langchain_aws import BedrockEmbeddings
from rag_app.vector_store import get_collection, ingest_chunks, query_collection

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def ensure_pdf_is_indexed(collection, embeddings_model) -> None:
    if collection.count() > 0:
        print(f"Using existing Chroma collection with {collection.count()} chunks.")
        return

    print("Collection is empty. Loading PDF and creating Titan embeddings...")

    # Load PDF
    pdf_loader = PyPDFLoader(PDF_PATH)
    docs = pdf_loader.load()

    # Split pdg into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )


    # Split documents into chunks
    chunk_docs = text_splitter.split_documents(docs)
    chunk_texts = [doc.page_content for doc in chunk_docs]

    print(f"Loaded PDF with {len(docs)} pages")
    print(f"Split into {len(chunk_texts)} chunks")

    embedding_vectors = embeddings_model.embed_documents(chunk_texts)

    ingest_chunks(
        collection=collection,
        chunks=chunk_texts,
        embeddings=embedding_vectors,
        id_prefix="tesla_earning_chunk",
    )

    print(f"Inserted {len(chunk_texts)} chunks into Chroma.")


def print_match_scores(results) -> None:
    for index, distance in enumerate(results["distances"][0], start=1):
        similarity = 1 - distance
        print(f"Match {index} similarity: {similarity:.4f}")


def main() -> None:
    collection = get_collection(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    bedrock = ChatBedrockConverse(
        model_id=MODEL_ID,
        region_name=AWS_REGION,
        max_tokens=700,
        temperature=0.2,
    )

    embeddings_model = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={
            "dimensions": 1024,
            "normalize": True,
        },
    )

    ensure_pdf_is_indexed(collection, embeddings_model)

    query = DEFAULT_QUERY

    query_embedding = embeddings_model.embed_query(query)

    results = query_collection(collection, query_embedding, n_results=3)

    matched_documents = results["documents"][0]

    context = "\n\n---\n\n".join(matched_documents)


    prompt = f"""Use the context below to answer the question.
        If the answer is not in the context, say you don't know.
        Context:
        {context}
        Question:
        {query}
        """


    # print_match_scores(results)


    response = bedrock.invoke(prompt)

    print("Question:")
    print(query)
    print("\nReply:")
    print(response.content)
