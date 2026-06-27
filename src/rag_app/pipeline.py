from typing import TypedDict

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
from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv

from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# LLM model
bedrock = ChatBedrockConverse(
    model_id=MODEL_ID,
    region_name=AWS_REGION,
    max_tokens=700,
    temperature=0.2,
)

class Agent(TypedDict):
    question: str
    rag_answer: str
    use_web_search: bool
    web_results: list[str]
    answer: str

def in_transcript(state: Agent) -> Agent:

    prompt = f"""
        You are judging whether a RAG answer actually answered the user's question.

        User question:
        {state["question"]}

        RAG answer:
        {state["rag_answer"]}

        If the RAG answer says it does not know, lacks context, cannot answer from the transcript,
        or does not directly answer the question, return:
        needs_web_search

        If the RAG answer directly answers the question, return:
        answered

        Return only:
        answered
        or
        needs_web_search
        """
    answer = bedrock.invoke(prompt)
    decision = answer.content.strip().lower()
    
    return {**state, "use_web_search": decision != "answered"}

def route_after_judge(state: Agent) -> str:
    if state["use_web_search"]:
        return "search_web"
    return "rag_answered"

def rag_answered(state: Agent) -> Agent:
    return {**state, "answer": state["rag_answer"]}

def search_web(state: Agent) -> Agent:
    content = []
    response = tavily.search(state["question"], max_results=5)
    for r in response["results"]:
        content.append(r["content"])
    return {**state, "web_results": content}

def final_answer(state: Agent) -> Agent:
    question = state["question"]
    web_results = state["web_results"]

    prompt = f"""
    Question:
    {question}

    Web Results:
    {web_results}

    I want to answer to this questions using the Web Results. If the answer isn't in the context, then
    reply I don't know.
    """

    response = bedrock.invoke(prompt)
    return {**state, "answer": response.content}

# Agent configuration
builder = StateGraph(Agent)
builder.add_node("in_transcript", in_transcript)
builder.add_node("rag_answered", rag_answered)
builder.add_node("search_web", search_web)
builder.add_node("final_answer", final_answer)

# Agent Edges
builder.set_entry_point("in_transcript")

builder.add_conditional_edges(
    "in_transcript",
    route_after_judge,
    {"rag_answered": "rag_answered", "search_web": "search_web"}
)
builder.add_edge("rag_answered", END)
builder.add_edge("search_web", "final_answer")
builder.add_edge("final_answer", END)

agent = builder.compile()

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
        chunk_size=1000,
        chunk_overlap=120,
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

    # Create a vector DB
    collection = get_collection(
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
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

    print_match_scores(results)

    prompt = f"""Use the context below to answer the question.
        If the answer is not in the context, say you don't know.
        Context:
        {context}
        Question:
        {query}
        """
    
    rag_answer = bedrock.invoke(prompt)
    
    response = agent.invoke({
            "question": DEFAULT_QUERY,
            "rag_answer": rag_answer.content
        }
    )

    print("Question:")
    print(query)
    print("\nReply:")
    print(response["answer"])
