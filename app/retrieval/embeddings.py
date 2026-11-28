from typing import cast

import requests

from app.core.config import settings


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{settings.ollama_url}/api/embed",
        json={
            "model": settings.embed_model,
            "input": text,
            "keep_alive": "5m",
        },
        timeout=settings.embedding_timeout,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise ValueError("Ollama returned an empty embedding")

    return cast(list[float], embeddings[0])
