import logging

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    HealthResponse,
    ModelsResponse,
)
from app.retrieval.vector_store import collection
from app.services.ollama import OllamaError, OllamaService
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()

ollama_service = OllamaService()
rag_service = RAGService(ollama=ollama_service)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check application health",
)
def health_check() -> HealthResponse:
    """Check application and Ollama availability."""

    ollama_available = ollama_service.health_check()

    return HealthResponse(
        status="ok" if ollama_available else "degraded",
        ollama="available" if ollama_available else "unavailable",
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
)
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve relevant documents and generate an answer."""

    try:
        result = rag_service.ask(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
        )

        return AskResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except OllamaError as exc:
        logger.exception("Ollama service failed")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The language model is currently unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("RAG request failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the question.",
        ) from exc


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available Ollama models",
)
def list_models() -> ModelsResponse:
    """Return models available on the Ollama server."""

    try:
        models = ollama_service.list_models()

        return ModelsResponse(
            models=models,
            count=len(models),
        )

    except OllamaError as exc:
        logger.exception("Unable to retrieve Ollama models")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve Ollama models.",
        ) from exc


@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List indexed documents",
)
def list_documents() -> DocumentsResponse:
    """Return documents indexed in ChromaDB."""

    try:
        records = collection.get(include=["metadatas"])

        documents = {}

        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue

            document_id = metadata.get("document_id")
            source = metadata.get("source", "Unknown")
            identifier = document_id or source

            if identifier not in documents:
                documents[identifier] = {
                    "document_id": document_id,
                    "source": source,
                    "source_path": metadata.get("source_path"),
                    "content_hash": metadata.get("content_hash"),
                    "chunks": 0,
                }

            documents[identifier]["chunks"] += 1

        document_list = list(documents.values())

        return DocumentsResponse(
            documents=document_list,
            count=len(document_list),
        )

    except Exception as exc:
        logger.exception("Unable to retrieve indexed documents")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve indexed documents.",
        ) from exc
