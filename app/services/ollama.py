import logging
from typing import Any

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Base exception for Ollama service errors."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama is unavailable or unreachable."""


class OllamaService:
    """Centralized client for the Ollama API."""

    def __init__(self):
        self.base_url = settings.ollama_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.generation_timeout

    def _request(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """Send an HTTP request to the Ollama API."""

        url = f"{self.base_url}/api/{endpoint}"

        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            return response.json()

        except requests.Timeout as exc:
            logger.error("Ollama request timed out: %s", exc)
            raise OllamaConnectionError("Ollama request timed out") from exc

        except requests.ConnectionError as exc:
            logger.error("Unable to connect to Ollama: %s", exc)
            raise OllamaConnectionError("Unable to connect to Ollama") from exc

        except requests.HTTPError as exc:
            logger.error("Ollama HTTP error: %s", exc)
            raise OllamaError(f"Ollama returned HTTP {response.status_code}") from exc

        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise OllamaError("Ollama request failed") from exc

        except ValueError as exc:
            logger.error("Invalid Ollama response: %s", exc)
            raise OllamaError("Ollama returned an invalid JSON response") from exc

    def _generation_options(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build generation options from application settings."""

        options = {
            "num_ctx": settings.context_length,
            "num_predict": (
                max_tokens if max_tokens is not None else settings.generation_max_tokens
            ),
            "num_thread": settings.num_threads,
        }

        if temperature is not None:
            options["temperature"] = temperature

        return options

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text using the configured Ollama model."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": self._generation_options(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        }

        if system:
            payload["system"] = system

        result = self._request(
            endpoint="generate",
            payload=payload,
        )

        return result.get("response", "")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a conversational response."""

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "5m",
            "options": self._generation_options(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        }

        result = self._request(
            endpoint="chat",
            payload=payload,
        )

        return result.get("message", {}).get("content", "")

    def health_check(self) -> bool:
        """Check whether the Ollama server is available."""

        try:
            self._request(
                endpoint="tags",
                method="GET",
            )
            return True

        except OllamaError:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """Return the models available on the Ollama server."""

        result = self._request(
            endpoint="tags",
            method="GET",
        )

        return result.get("models", [])
