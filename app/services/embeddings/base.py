"""Abstract base class definition for embedding services.

Defines the interface for all text embedding providers (Local, OpenAI, Gemini)
with asynchronous single and batch embedding generation.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    """Abstract base class defining the interface for text embedding providers.

    All embedding providers must implement dimension reporting, provider naming,
    single-text embedding, and batch embedding asynchronously.
    """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of vectors produced by this embedding provider.

        Returns:
            int: Vector dimension count (e.g., 384 for all-MiniLM-L6-v2, 1536 for OpenAI).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier name of the embedding provider.

        Returns:
            str: Provider name identifier (e.g., 'local', 'openai', 'gemini').
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate a normalized vector embedding for a single text string.

        Args:
            text: Input text string to embed.

        Returns:
            list[float]: Embedding vector with length matching self.dimensions.

        Raises:
            GatewayError: If embedding generation fails.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized vector embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors, one for each input text.

        Raises:
            GatewayError: If batch embedding generation fails.
        """
        ...
