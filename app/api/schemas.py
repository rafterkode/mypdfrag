from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for document-based questions."""

    question: str = Field(
        min_length=1,
        description="Question to ask about indexed documents.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Number of document chunks to retrieve.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature.",
    )


class SourceReference(BaseModel):
    """Reference to a source document."""

    source: str
    page: str | int


class AskResponse(BaseModel):
    """Response model for RAG queries."""

    question: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Application health status."""

    status: str
    ollama: str


class ModelsResponse(BaseModel):
    """Available Ollama models."""

    models: list[dict[str, Any]]
    count: int


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    document_id: str | None = None
    source: str
    source_path: str | None = None
    content_hash: str | None = None
    chunks: int = 0


class DocumentsResponse(BaseModel):
    """List of indexed documents."""

    documents: list[DocumentInfo]
    count: int
