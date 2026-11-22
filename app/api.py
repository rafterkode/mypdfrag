## API for request on database
import time
from typing import Any, cast

import chromadb
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MyPDFRag API")

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "embeddinggemma:300m-qat-q4_0"

DEBUG = False

client = chromadb.PersistentClient(path="./database")

collection = client.get_or_create_collection(name="documents")


class Question(BaseModel):
    question: str


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text, "keep_alive": "5m"},
        timeout=180,
    )

    response.raise_for_status()
    return cast(list[float], response.json()["embeddings"][0])


@app.post("/ask")
def ask(data: Question) -> dict[str, Any]:

    start_time = time.perf_counter()

    # 1. Generate question embedding
    step_start = time.perf_counter()

    query_embedding = generate_embedding(data.question)

    if DEBUG:
        print(f"Embedding: {time.perf_counter() - step_start:.2f}s")

    # 2. Retrieve relevant document chunks
    step_start = time.perf_counter()

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if DEBUG:
        print(f"ChromaDB: {time.perf_counter() - step_start:.2f}s")

    if not results["documents"] or not results["metadatas"]:
        return {
            "question": data.question,
            "answer": "No documents found.",
            "sources": [],
        }

    retrieved_chunks = results["documents"][0]
    source_metadata = results["metadatas"][0]

    retrieved_context = "\n\n".join(
        [
            f"[Source: {source_info['source']}, page {source_info['page']}]\n{chunk}"
            for chunk, source_info in zip(retrieved_chunks, source_metadata)
        ]
    )

    # 3. Construct the prompt
    prompt = f"""
You are an assistant specialized in document analysis.

Your task is to answer questions using exclusively the information
provided in the retrieved context.

Rules:
- Answer in the same language as the user's question.
- French and English are the primary supported languages.
- You may use information from documents written in a different
  language than the user's question.
- Use only the information provided in the context.
- Never fabricate information or fill in missing details
  using your general knowledge.
- If the context does not contain enough information to answer,
  clearly state that.
- Provide clear, natural, and well-structured answers.
- Avoid unnecessary repetition and irrelevant information.
- Mention sources only when they add meaningful value to the answer.

CONTEXT:
{retrieved_context}

QUESTION:
{data.question}

ANSWER:
"""

    # 4. Generate response
    step_start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "num_ctx": 2048,
                "num_predict": 500,
                "num_thread": 4,
                "temperature": 0.1,
            },
        },
        timeout=300,
    )

    response.raise_for_status()
    generation_result = response.json()

    if DEBUG:
        print(f"Generation: {time.perf_counter() - step_start:.2f}s")
        print(f"Total time: {time.perf_counter() - start_time:.2f}s")
        print(f"Generation stop reason: {generation_result.get('done_reason')}")

    # 5. Return the response
    return {
        "question": data.question,
        "answer": generation_result["response"],
        "sources": source_metadata,
    }
