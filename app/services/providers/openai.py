"""OpenAI LLM provider adapter implementation.

Translates unified ChatCompletionRequest payloads into OpenAI API calls via AsyncOpenAI,
and maps responses and streaming events back into normalized proxy schemas.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import openai
import structlog

from app.api.v1.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    DeltaContent,
    UsageInfo,
)
from app.core.config import Settings
from app.core.exceptions import ProviderAPIError, RateLimitError
from app.services.providers.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Upstream provider adapter for OpenAI models (GPT-4o, GPT-4o-mini, etc.).

    Integrates using the official OpenAI AsyncOpenAI Python SDK.
    """

    provider_name: str = "openai"

    def __init__(self, settings: Settings) -> None:
        """Initialize the OpenAI provider adapter.

        Args:
            settings: Core application settings containing API keys and timeouts.
        """
        self._settings = settings
        api_key = settings.OPENAI_API_KEY.strip() if settings.OPENAI_API_KEY else "sk-dummy-key"
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=float(settings.CB_REQUEST_TIMEOUT_MS) / 1000.0,
            max_retries=0,  # Proxy handles retry and failover policies
        )
        logger.info("Initialized OpenAIProvider adapter", has_key=bool(settings.OPENAI_API_KEY))

    def _map_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Format domain messages into OpenAI API compatible message dictionaries.

        Args:
            messages: List of ChatMessage objects.

        Returns:
            list[dict[str, str]]: Serialized messages conforming to OpenAI specification.
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _handle_exception(self, exc: Exception) -> Exception:
        """Map upstream OpenAI exceptions to normalized Gateway domain exceptions.

        Args:
            exc: Exception raised by the OpenAI SDK.

        Returns:
            Exception: Domain-specific GatewayError subclass.
        """
        if isinstance(exc, openai.RateLimitError):
            retry_after: float | None = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_header = exc.response.headers.get(
                    "retry-after-ms"
                ) or exc.response.headers.get("retry-after")
                if retry_header:
                    try:
                        retry_after = (
                            float(retry_header) / 1000.0
                            if "ms" in str(retry_header)
                            else float(retry_header)
                        )
                    except (ValueError, TypeError):
                        retry_after = None

            logger.warning(
                "OpenAI upstream rate limit exceeded",
                error=str(exc),
                retry_after=retry_after,
            )
            return RateLimitError(
                provider_name=self.provider_name,
                retry_after=retry_after,
                message=f"OpenAI rate limit reached: {exc.message if hasattr(exc, 'message') else str(exc)}",
                details={"error_body": getattr(exc, "body", None)},
            )

        if isinstance(exc, openai.APIStatusError):
            status_code = exc.status_code
            logger.error(
                "OpenAI upstream API status error",
                status_code=status_code,
                error=str(exc),
            )
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=status_code,
                message=f"OpenAI returned HTTP {status_code}: {exc.message if hasattr(exc, 'message') else str(exc)}",
                details={"response_body": getattr(exc, "body", None)},
            )

        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            logger.error("OpenAI connection or timeout error", error=str(exc))
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=504,
                message=f"OpenAI connection error: {str(exc)}",
                details={"type": type(exc).__name__},
            )

        if isinstance(exc, openai.OpenAIError):
            logger.error("OpenAI generic SDK error", error=str(exc))
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=502,
                message=f"OpenAI SDK error: {str(exc)}",
                details={"type": type(exc).__name__},
            )

        logger.error("Unexpected error in OpenAI provider", error=str(exc))
        return ProviderAPIError(
            provider_name=self.provider_name,
            status_code=500,
            message=f"Unexpected error communicating with OpenAI: {str(exc)}",
        )

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a complete chat completion via OpenAI API.

        Args:
            request: Standardized completion request.

        Returns:
            ChatCompletionResponse: Normalized chat completion response.

        Raises:
            RateLimitError: If OpenAI returns HTTP 429.
            ProviderAPIError: For other OpenAI upstream errors.
        """
        model_name = request.model or "gpt-4o"
        messages = self._map_messages(request.messages)

        logger.debug(
            "Dispatching OpenAI completion request",
            model=model_name,
            message_count=len(messages),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        try:
            response: Any = await self._client.chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )

            choices = [
                Choice(
                    index=c.index,
                    message=ChatMessage(
                        role=c.message.role or "assistant",
                        content=c.message.content or "",
                    ),
                    finish_reason=c.finish_reason,
                )
                for c in response.choices
            ]

            usage = None
            if response.usage:
                usage = UsageInfo(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return ChatCompletionResponse(
                id=response.id,
                object="chat.completion",
                created=response.created or int(time.time()),
                choices=choices,
                model=response.model or model_name,
                usage=usage,
                cached=False,
            )

        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream completion tokens from OpenAI API as unified SSE chunks.

        Args:
            request: Standardized completion request with streaming enabled.

        Yields:
            ChatCompletionChunk: Normalized streaming response chunks.

        Raises:
            RateLimitError: If OpenAI rate limits during stream initiation.
            ProviderAPIError: If an error occurs during stream transmission.
        """
        model_name = request.model or "gpt-4o"
        messages = self._map_messages(request.messages)

        logger.debug(
            "Dispatching OpenAI stream request",
            model=model_name,
            message_count=len(messages),
        )

        try:
            stream_resp = await self._client.chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )
        except Exception as exc:
            raise self._handle_exception(exc) from exc

        try:
            async for chunk in stream_resp:
                choices = [
                    Choice(
                        index=c.index,
                        delta=DeltaContent(
                            role=c.delta.role,
                            content=c.delta.content,
                        ),
                        finish_reason=c.finish_reason,
                    )
                    for c in chunk.choices
                ]

                yield ChatCompletionChunk(
                    id=chunk.id,
                    object="chat.completion.chunk",
                    created=chunk.created or int(time.time()),
                    choices=choices,
                    model=chunk.model or model_name,
                )
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def health_check(self) -> bool:
        """Perform health check by verifying model listing capability.

        Returns:
            bool: True if OpenAI endpoint is responsive, False on connection/auth errors.
        """
        try:
            # Quick lightweight check
            await self._client.models.list(timeout=3.0)
            return True
        except Exception as exc:
            logger.warning("OpenAI health check failed", error=str(exc))
            return False

    async def close(self) -> None:
        """Close the underlying AsyncOpenAI HTTP client connection pool."""
        try:
            await self._client.close()
            logger.debug("Closed OpenAI client session")
        except Exception as exc:
            logger.warning("Error closing OpenAI client", error=str(exc))
