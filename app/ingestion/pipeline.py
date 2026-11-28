import hashlib
from pathlib import Path

from app.ingestion.pdf import extract_chunks
from app.retrieval.embeddings import generate_embedding
from app.retrieval.vector_store import collection


def index_pdf(pdf_path: str | Path) -> int:
    """
    Extract, embed and index a PDF document.

    Existing chunks are replaced only after all new embeddings
    have been successfully generated.
    """

    pdf_path = Path(pdf_path).resolve()

    document_chunks = extract_chunks(pdf_path)

    if not document_chunks:
        raise ValueError(f"No extractable text found in PDF: {pdf_path.name}")

    # Stable logical document identifier.
    document_id = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()

    # Detect document content changes.
    content_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []

    # Generate all embeddings before modifying ChromaDB.
    for chunk_index, chunk in enumerate(document_chunks):
        chunk_id = hashlib.sha256(
            f"{document_id}:{content_hash}:{chunk_index}".encode()
        ).hexdigest()

        embedding = generate_embedding(chunk["text"])

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk["text"])

        metadatas.append(
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "source": chunk["source"],
                "source_path": str(pdf_path),
                "page": chunk["page"],
            }
        )

    # Insert or update the new chunks.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    # Find previous chunks belonging to this source.
    previous_records = collection.get(
        where={"source": pdf_path.name},
        include=["metadatas"],
    )

    # Delete obsolete chunks, keeping the newly indexed ones.
    obsolete_ids = [
        record_id for record_id in previous_records["ids"] if record_id not in ids
    ]

    if obsolete_ids:
        collection.delete(ids=obsolete_ids)

    print(f"Indexing completed: {pdf_path.name} ({len(document_chunks)} chunks)")

    return len(document_chunks)
