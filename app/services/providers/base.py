"""Abstract base class definition for LLM provider adapters.

Defines the standard interface for upstream provider interactions including
synchronous generation, streaming, health verification, and resource disposal.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.api.v1.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)


class BaseLLMProvider(ABC):
    """Abstract base class establishing the contract for all LLM provider adapters.

    Every concrete provider (e.g. OpenAI, Anthropic, Gemini, Mock) implements this
    interface to ensure uniform consumption by the routing and proxy layers.

    Attributes:
        provider_name: Canonical string identifier for the provider (e.g., 'mock', 'openai').
    """

    provider_name: str = "base"

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a complete non-streaming chat completion response.

        Args:
            request: Unified chat completion request containing messages and parameters.

        Returns:
            ChatCompletionResponse: Standardized response with generated content and usage.

        Raises:
            ProviderAPIError: If the upstream provider returns an error.
            RateLimitError: If the upstream provider rate limits the request.
        """
        ...

    @abstractmethod
    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream response tokens as OpenAI-compatible Server-Sent Event (SSE) chunks.

        Args:
            request: Unified chat completion request with streaming enabled.

        Yields:
            ChatCompletionChunk: Incremental delta chunks with partial content and metadata.

        Raises:
            ProviderAPIError: If the upstream streaming connection encounters an error.
            RateLimitError: If upstream rate limit is encountered during stream setup.
        """
        ...

    async def health_check(self) -> bool:
        """Perform a readiness and liveness check against the provider service.

        Returns:
            bool: True if the provider is healthy and operational, False otherwise.
        """
        return True

    async def close(self) -> None:
        """Release underlying HTTP client sessions and connection pools.

        Default implementation is a no-op; overriding subclasses should clean up network resources.
        """
        return None
