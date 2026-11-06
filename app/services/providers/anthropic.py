"""Anthropic Claude LLM provider adapter implementation.

Translates unified ChatCompletionRequest payloads into Anthropic Messages API calls,
separating system instructions from message history and converting streaming events
into normalized OpenAI-compatible chunks.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic
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


class AnthropicProvider(BaseLLMProvider):
    """Upstream provider adapter for Anthropic Claude models (Claude 3.5 Sonnet, Claude 3 Haiku, etc.).

    Integrates using the official Anthropic AsyncAnthropic Python SDK.
    """

    provider_name: str = "anthropic"

    _STOP_REASON_MAP: dict[str, str] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }

    def __init__(self, settings: Settings) -> None:
        """Initialize the Anthropic provider adapter.

        Args:
            settings: Core application settings containing API keys and timeouts.
        """
        self._settings = settings
        api_key = (
            settings.ANTHROPIC_API_KEY.strip() if settings.ANTHROPIC_API_KEY else "sk-ant-dummy-key"
        )
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=float(settings.CB_REQUEST_TIMEOUT_MS) / 1000.0,
            max_retries=0,  # Proxy handles retry and failover policies
        )
        logger.info(
            "Initialized AnthropicProvider adapter", has_key=bool(settings.ANTHROPIC_API_KEY)
        )

    def _prepare_payload(
        self, request: ChatCompletionRequest
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Extract system prompt and format messages for Anthropic Messages API.

        Anthropic requires system instructions to be passed as a top-level parameter
        rather than inside the conversation messages list.

        Args:
            request: Unified chat completion request.

        Returns:
            tuple[str | None, list[dict[str, str]]]: (system_instruction, messages)
        """
        system_chunks: list[str] = []
        messages: list[dict[str, str]] = []

        for msg in request.messages:
            if msg.role == "system":
                system_chunks.append(msg.content)
            elif msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})

        system_instruction = "\n\n".join(system_chunks) if system_chunks else None

        # Anthropic requires at least one message in the conversation
        if not messages:
            messages = [{"role": "user", "content": system_instruction or "Hello"}]
            system_instruction = None

        return system_instruction, messages

    def _map_stop_reason(self, stop_reason: str | None) -> str | None:
        """Map Anthropic stop reasons to OpenAI-compatible finish_reasons.

        Args:
            stop_reason: Anthropic stop reason string.

        Returns:
            str | None: Normalized finish reason (e.g. 'stop', 'length').
        """
        if not stop_reason:
            return None
        return self._STOP_REASON_MAP.get(stop_reason, stop_reason)

    def _handle_exception(self, exc: Exception) -> Exception:
        """Map upstream Anthropic exceptions to Gateway domain exceptions.

        Args:
            exc: Exception raised by the Anthropic SDK.

        Returns:
            Exception: Domain-specific GatewayError subclass.
        """
        if isinstance(exc, anthropic.RateLimitError):
            retry_after: float | None = None
            if hasattr(exc, "response") and exc.response is not None:
                header = exc.response.headers.get("retry-after")
                if header:
                    try:
                        retry_after = float(header)
                    except (ValueError, TypeError):
                        retry_after = None

            logger.warning(
                "Anthropic upstream rate limit exceeded",
                error=str(exc),
                retry_after=retry_after,
            )
            return RateLimitError(
                provider_name=self.provider_name,
                retry_after=retry_after,
                message=f"Anthropic rate limit reached: {str(exc)}",
                details={"error_body": getattr(exc, "body", None)},
            )

        if isinstance(exc, anthropic.APIStatusError):
            status_code = exc.status_code
            logger.error(
                "Anthropic upstream API status error",
                status_code=status_code,
                error=str(exc),
            )
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=status_code,
                message=f"Anthropic returned HTTP {status_code}: {str(exc)}",
                details={"response_body": getattr(exc, "body", None)},
            )

        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            logger.error("Anthropic connection or timeout error", error=str(exc))
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=504,
                message=f"Anthropic connection error: {str(exc)}",
                details={"type": type(exc).__name__},
            )

        if isinstance(exc, anthropic.AnthropicError):
            logger.error("Anthropic generic SDK error", error=str(exc))
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=502,
                message=f"Anthropic SDK error: {str(exc)}",
                details={"type": type(exc).__name__},
            )

        logger.error("Unexpected error in Anthropic provider", error=str(exc))
        return ProviderAPIError(
            provider_name=self.provider_name,
            status_code=500,
            message=f"Unexpected error communicating with Anthropic: {str(exc)}",
        )

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a complete chat completion via Anthropic Messages API.

        Args:
            request: Standardized completion request.

        Returns:
            ChatCompletionResponse: Normalized chat completion response.

        Raises:
            RateLimitError: If Anthropic returns HTTP 429.
            ProviderAPIError: For other Anthropic upstream errors.
        """
        model_name = request.model or "claude-3-5-sonnet-20241022"
        system_instruction, messages = self._prepare_payload(request)

        # Anthropic requires max_tokens to be specified
        max_tokens = request.max_tokens or 1024

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        logger.debug(
            "Dispatching Anthropic completion request",
            model=model_name,
            message_count=len(messages),
            has_system=bool(system_instruction),
        )

        try:
            response = await self._client.messages.create(**kwargs)

            # Concatenate all text content blocks
            content_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

            finish_reason = self._map_stop_reason(response.stop_reason)

            usage = None
            if response.usage:
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

            return ChatCompletionResponse(
                id=response.id,
                object="chat.completion",
                created=int(time.time()),
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=content_text,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                model=response.model or model_name,
                usage=usage,
                cached=False,
            )

        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream completion tokens from Anthropic Messages API as unified SSE chunks.

        Args:
            request: Standardized completion request with streaming enabled.

        Yields:
            ChatCompletionChunk: Normalized streaming response chunks.

        Raises:
            RateLimitError: If Anthropic rate limits during stream setup.
            ProviderAPIError: If an error occurs during stream transmission.
        """
        model_name = request.model or "claude-3-5-sonnet-20241022"
        system_instruction, messages = self._prepare_payload(request)
        max_tokens = request.max_tokens or 1024

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        logger.debug(
            "Dispatching Anthropic stream request",
            model=model_name,
            message_count=len(messages),
        )

        try:
            stream_resp = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise self._handle_exception(exc) from exc

        message_id = f"msg_{int(time.time())}"
        created_ts = int(time.time())

        try:
            async for event in stream_resp:
                event_type = getattr(event, "type", "")

                if event_type == "message_start":
                    if hasattr(event, "message"):
                        message_id = getattr(event.message, "id", message_id)
                        model_name = getattr(event.message, "model", model_name)

                    # Initial role delta chunk
                    yield ChatCompletionChunk(
                        id=message_id,
                        object="chat.completion.chunk",
                        created=created_ts,
                        choices=[
                            Choice(
                                index=0,
                                delta=DeltaContent(role="assistant", content=""),
                                finish_reason=None,
                            )
                        ],
                        model=model_name,
                    )

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_text = getattr(delta, "text", "") if delta else ""
                    if delta_text:
                        yield ChatCompletionChunk(
                            id=message_id,
                            object="chat.completion.chunk",
                            created=created_ts,
                            choices=[
                                Choice(
                                    index=0,
                                    delta=DeltaContent(role=None, content=delta_text),
                                    finish_reason=None,
                                )
                            ],
                            model=model_name,
                        )

                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None) if delta else None
                    finish_reason = self._map_stop_reason(stop_reason) or "stop"
                    yield ChatCompletionChunk(
                        id=message_id,
                        object="chat.completion.chunk",
                        created=created_ts,
                        choices=[
                            Choice(
                                index=0,
                                delta=DeltaContent(role=None, content=None),
                                finish_reason=finish_reason,
                            )
                        ],
                        model=model_name,
                    )

        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def health_check(self) -> bool:
        """Perform health check on Anthropic adapter connectivity.

        Returns:
            bool: True if client is initialized and operational.
        """
        return bool(self._client)

    async def close(self) -> None:
        """Close the underlying AsyncAnthropic HTTP client connection pool."""
        try:
            await self._client.close()
            logger.debug("Closed Anthropic client session")
        except Exception as exc:
            logger.warning("Error closing Anthropic client", error=str(exc))
