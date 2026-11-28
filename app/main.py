import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.services.ollama import OllamaService

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    logger.info("Starting application")

    ollama = OllamaService()

    if ollama.health_check():
        logger.info("Ollama connection established")
    else:
        logger.warning("Ollama is currently unavailable")

    yield

    logger.info("Shutting down application")


app = FastAPI(
    title="MyPDFRAG API",
    description="Local document-based question answering API.",
    version="2.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(
    router,
    prefix="/api",
    tags=["RAG"],
)


@app.get("/", tags=["General"])
def root():
    """Return basic API information."""

    return {
        "name": "MyPDFRAG API",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
    }
