from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Ollama
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    embed_model: str = "embeddinggemma:300m-qat-q4_0"

    # Storage
    database_path: str = str(PROJECT_ROOT / "database")
    documents_path: str = str(PROJECT_ROOT / "documents")
    collection_name: str = "documents"

    # Retrieval
    retrieval_top_k: int = 3

    # PDF chunking
    chunk_size: int = 1500
    chunk_overlap: int = 250
    min_chunk_chars: int = 100

    # Timeouts
    embedding_timeout: int = 180
    generation_timeout: int = 300

    # LLM generation
    context_length: int = 2048
    generation_max_tokens: int = 500
    num_threads: int = 4

    # Application
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
