import logging
from typing import Any

from app.core.config import settings
from app.retrieval.embeddings import generate_embedding
from app.retrieval.vector_store import collection
from app.services.ollama import OllamaService

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrates document retrieval and answer generation."""

    SYSTEM_PROMPT = """
You are a document-based question answering assistant.

Your task is to answer questions using only the
provided documentary context.

Rules:
- Use only information from the provided context.
- Never invent facts or information.
- If the context does not contain the answer,
  explicitly state that you do not know.
- Reference the source documents when possible.
- Answer in the same language as the user's question.
- Provide clear and concise answers.
"""

    def __init__(
        self,
        ollama: OllamaService | None = None,
    ):
        self.ollama = ollama or OllamaService()
        self.top_k = settings.retrieval_top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks from ChromaDB."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        embedding = generate_embedding(question)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k or self.top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        retrieved_documents = []

        for index, content in enumerate(documents):
            retrieved_documents.append(
                {
                    "content": content,
                    "metadata": (
                        metadatas[index]
                        if index < len(metadatas) and metadatas[index]
                        else {}
                    ),
                    "distance": (distances[index] if index < len(distances) else None),
                }
            )

        return retrieved_documents

    @staticmethod
    def build_context(
        documents: list[dict[str, Any]],
    ) -> str:
        """Build the documentary context for the LLM."""

        context_parts = []

        for index, document in enumerate(documents, start=1):
            content = document.get("content", "")
            metadata = document.get("metadata", {})

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            context_parts.append(f"[Source {index}: {source}, page {page}]\n{content}")

        return "\n\n".join(context_parts)

    @staticmethod
    def extract_sources(
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract unique source references from retrieved chunks."""

        sources = []
        seen = set()

        for document in documents:
            metadata = document.get("metadata", {})

            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            source_key = (source, page)

            if source_key in seen:
                continue

            seen.add(source_key)

            sources.append(
                {
                    "source": source,
                    "page": page,
                }
            )

        return sources

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant context and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info("Processing RAG query")

        documents = self.retrieve(
            question=question,
            top_k=top_k,
        )

        if not documents:
            return {
                "question": question,
                "answer": ("No relevant documents were found to answer this question."),
                "sources": [],
            }

        context = self.build_context(documents)

        user_prompt = f"""
Documentary context:

{context}

User question:

{question}

Provide an answer based strictly on the documentary context.
"""

        answer = self.ollama.chat(
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": self.extract_sources(documents),
        }
