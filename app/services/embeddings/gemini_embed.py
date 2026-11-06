"""Google Gemini embedding service implementation.

Provides asynchronous text embedding generation using Google GenAI SDK
and models such as text-embedding-004 (768 dims).
"""

import asyncio
from typing import Any

from app.core.exceptions import ProviderAPIError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingService

logger = get_logger(__name__)


class GeminiEmbeddingService(BaseEmbeddingService):
    """Gemini embedding service for generating embeddings via Google GenAI API."""

    _default_dimensions: int = 768
    _provider_name: str = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-004",
        dimensions: int = 768,
        client: Any | None = None,
    ) -> None:
        """Initialize Gemini embedding service.

        Args:
            api_key: Google GenAI API key.
            model_name: Google embedding model name.
            dimensions: Expected embedding dimensions (default 768).
            client: Optional pre-configured GenAI client instance.
        """
        self.api_key = api_key or ""
        self.model_name = model_name
        self._dimensions = dimensions
        self._client = client

    @property
    def dimensions(self) -> int:
        """Vector dimension count (768 for text-embedding-004)."""
        return self._dimensions

    @property
    def provider_name(self) -> str:
        """Provider identifier string ('gemini')."""
        return self._provider_name

    def _get_client(self) -> Any:
        """Lazily initialize Google GenAI client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key or "mock-key")
            except Exception as e:
                logger.error("gemini_client_init_failed", error=str(e))
                raise ProviderAPIError(
                    provider_name=self.provider_name,
                    message=f"Failed to initialize Google GenAI client: {e}",
                ) from e
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Text string to embed.

        Returns:
            list[float]: Embedding vector with length self.dimensions.

        Raises:
            ProviderAPIError: If the Google GenAI API request fails.
        """
        client = self._get_client()
        try:
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                response = await client.aio.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
            else:
                response = await asyncio.to_thread(
                    client.models.embed_content,
                    model=self.model_name,
                    contents=text,
                )

            # Parse embedding values from Google GenAI response
            if hasattr(response, "embedding") and response.embedding is not None:
                return list(response.embedding.values)
            if (
                hasattr(response, "embeddings")
                and response.embeddings
                and len(response.embeddings) > 0
            ):
                return list(response.embeddings[0].values)

            # Fallback dictionary access if response is dict-like
            if isinstance(response, dict):
                if "embedding" in response:
                    return list(response["embedding"]["values"])
                if "embeddings" in response and len(response["embeddings"]) > 0:
                    return list(response["embeddings"][0]["values"])

            raise ValueError("Unexpected response structure from Google GenAI embedding API")
        except Exception as e:
            logger.error("gemini_embed_failed", model=self.model_name, error=str(e))
            raise ProviderAPIError(
                provider_name=self.provider_name,
                message=f"Gemini embedding generation failed: {e}",
                details={"model": self.model_name},
            ) from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors matching input order.

        Raises:
            ProviderAPIError: If the Google GenAI API request fails.
        """
        if not texts:
            return []

        client = self._get_client()
        try:
            if hasattr(client, "aio") and hasattr(client.aio, "models"):
                response = await client.aio.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                )
            else:
                response = await asyncio.to_thread(
                    client.models.embed_content,
                    model=self.model_name,
                    contents=texts,
                )

            if hasattr(response, "embeddings") and response.embeddings:
                return [list(emb.values) for emb in response.embeddings]
            if isinstance(response, dict) and "embeddings" in response:
                return [list(emb["values"]) for emb in response["embeddings"]]

            # If response returned single embedding for list
            if hasattr(response, "embedding") and response.embedding is not None:
                return [list(response.embedding.values)]

            raise ValueError("Unexpected batch response structure from Google GenAI embedding API")
        except Exception as e:
            logger.error(
                "gemini_batch_embed_failed",
                model=self.model_name,
                batch_size=len(texts),
                error=str(e),
            )
            raise ProviderAPIError(
                provider_name=self.provider_name,
                message=f"Gemini batch embedding generation failed: {e}",
                details={"model": self.model_name, "batch_size": len(texts)},
            ) from e
