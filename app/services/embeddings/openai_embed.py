"""OpenAI embedding service implementation.

Provides asynchronous text embedding generation using OpenAI's AsyncOpenAI client
and models like text-embedding-3-small (1536 dims).
"""

from typing import Any

from app.core.exceptions import ProviderAPIError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingService

logger = get_logger(__name__)


class OpenAIEmbeddingService(BaseEmbeddingService):
    """OpenAI embedding service for generating embeddings via OpenAI API."""

    _default_dimensions: int = 1536
    _provider_name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        client: Any | None = None,
    ) -> None:
        """Initialize OpenAI embedding service.

        Args:
            api_key: OpenAI API key.
            model_name: OpenAI embedding model name.
            dimensions: Expected embedding dimensions (default 1536).
            client: Optional pre-configured AsyncOpenAI client instance.
        """
        self.api_key = api_key or ""
        self.model_name = model_name
        self._dimensions = dimensions
        self._client = client

    @property
    def dimensions(self) -> int:
        """Vector dimension count (1536 for text-embedding-3-small)."""
        return self._dimensions

    @property
    def provider_name(self) -> str:
        """Provider identifier string ('openai')."""
        return self._provider_name

    def _get_client(self) -> Any:
        """Lazily initialize AsyncOpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self.api_key or "mock-key")
            except Exception as e:
                logger.error("openai_client_init_failed", error=str(e))
                raise ProviderAPIError(
                    provider_name=self.provider_name,
                    message=f"Failed to initialize OpenAI client: {e}",
                ) from e
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Text string to embed.

        Returns:
            list[float]: Embedding vector with length self.dimensions.

        Raises:
            ProviderAPIError: If the OpenAI API request fails.
        """
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                input=text,
                model=self.model_name,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("openai_embed_failed", model=self.model_name, error=str(e))
            raise ProviderAPIError(
                provider_name=self.provider_name,
                message=f"OpenAI embedding generation failed: {e}",
                details={"model": self.model_name},
            ) from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors matching input order.

        Raises:
            ProviderAPIError: If the OpenAI API request fails.
        """
        if not texts:
            return []

        client = self._get_client()
        try:
            response = await client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            # Ensure ordering matches input indices
            sorted_items = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in sorted_items]
        except Exception as e:
            logger.error(
                "openai_batch_embed_failed",
                model=self.model_name,
                batch_size=len(texts),
                error=str(e),
            )
            raise ProviderAPIError(
                provider_name=self.provider_name,
                message=f"OpenAI batch embedding generation failed: {e}",
                details={"model": self.model_name, "batch_size": len(texts)},
            ) from e
