## Ingest documents on database
import hashlib
import os
from typing import Any

import chromadb
import pymupdf
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma:300m-qat-q4_0"

client = chromadb.PersistentClient(path="./database")

collection = client.get_or_create_collection(name="documents")


def extract_chunks(pdf_path: str) -> list[dict[str, Any]]:
    document = pymupdf.open(pdf_path)
    document_chunks: list[dict[str, Any]] = []

    for page_number, page in enumerate(document):
        page_text = page.get_text("text")

        # Split text into overlapping chunks
        chunk_size = 1500
        chunk_overlap = 250

        for start_position in range(0, len(page_text), chunk_size - chunk_overlap):
            chunk_text = page_text[start_position : start_position + chunk_size].strip()

            if len(chunk_text) > 100:
                document_chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number + 1,
                        "source": os.path.basename(pdf_path),
                    }
                )

    document.close()

    return document_chunks


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text, "keep_alive": "5m"},
        timeout=180,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def index_pdf(pdf_path: str) -> None:
    document_chunks = extract_chunks(pdf_path)

    for chunk_index, chunk in enumerate(document_chunks):
        document_id = hashlib.sha256(f"{pdf_path}:{chunk_index}".encode()).hexdigest()

        embedding = generate_embedding(chunk["text"])

        collection.upsert(
            ids=[document_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "page": chunk["page"]}],
        )

    print(f"Indexing completed: {pdf_path}")


if __name__ == "__main__":
    documents_directory = "./documents"

    for filename in os.listdir(documents_directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_directory, filename)

            index_pdf(pdf_path)
