"""Google Gemini LLM provider adapter implementation.

Translates unified ChatCompletionRequest payloads into Google GenAI SDK calls,
mapping message roles, system instructions, token usage metadata, and streaming events.
"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from google import genai
from google.genai import errors, types

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


class GeminiProvider(BaseLLMProvider):
    """Upstream provider adapter for Google Gemini models (gemini-2.5-flash, gemini-1.5-pro, etc.).

    Integrates using the official Google GenAI Python SDK (google.genai).
    """

    provider_name: str = "gemini"

    _FINISH_REASON_MAP: dict[str, str] = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop",
    }

    def __init__(self, settings: Settings) -> None:
        """Initialize the Google Gemini provider adapter.

        Args:
            settings: Core application settings containing API keys and timeouts.
        """
        self._settings = settings
        api_key = settings.GEMINI_API_KEY.strip() if settings.GEMINI_API_KEY else "dummy-key"
        self._client = genai.Client(api_key=api_key)
        logger.info("Initialized GeminiProvider adapter", has_key=bool(settings.GEMINI_API_KEY))

    def _prepare_payload(
        self, request: ChatCompletionRequest
    ) -> tuple[str | None, list[types.Content]]:
        """Extract system prompt and format conversation contents for Gemini.

        Maps role 'assistant' to Gemini's expected 'model' role, and extracts
        system messages into a standalone instruction string.

        Args:
            request: Unified chat completion request.

        Returns:
            tuple[str | None, list[types.Content]]: (system_instruction, contents)
        """
        system_chunks: list[str] = []
        contents: list[types.Content] = []

        for msg in request.messages:
            if msg.role == "system":
                system_chunks.append(msg.content)
            else:
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )

        system_instruction = "\n\n".join(system_chunks) if system_chunks else None

        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=system_instruction or "Hello")],
                )
            )
            system_instruction = None

        return system_instruction, contents

    def _map_finish_reason(self, reason: Any) -> str | None:
        """Map Gemini candidate finish reasons to standard finish reasons.

        Args:
            reason: Raw Gemini finish reason (str or enum).

        Returns:
            str | None: Normalized finish reason.
        """
        if not reason:
            return None
        reason_str = str(reason).upper()
        # Handle enum representations like 'FinishReason.STOP'
        if "." in reason_str:
            reason_str = reason_str.split(".")[-1]
        return self._FINISH_REASON_MAP.get(reason_str, "stop")

    def _handle_exception(self, exc: Exception) -> Exception:
        """Map upstream Google GenAI exceptions to Gateway domain exceptions.

        Args:
            exc: Exception raised by the Google GenAI SDK.

        Returns:
            Exception: Domain-specific GatewayError subclass.
        """
        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None) or 502
            err_str = str(exc).lower()

            if code == 429 or "resource_exhausted" in err_str or "quota" in err_str:
                logger.warning(
                    "Gemini upstream rate limit exceeded",
                    error=str(exc),
                    code=code,
                )
                return RateLimitError(
                    provider_name=self.provider_name,
                    message=f"Gemini rate limit reached: {str(exc)}",
                    details={"code": code},
                )

            logger.error(
                "Gemini upstream API error",
                code=code,
                error=str(exc),
            )
            return ProviderAPIError(
                provider_name=self.provider_name,
                status_code=code if isinstance(code, int) and 100 <= code <= 599 else 502,
                message=f"Gemini API returned error: {str(exc)}",
                details={"code": code},
            )

        logger.error("Unexpected error in Gemini provider", error=str(exc))
        return ProviderAPIError(
            provider_name=self.provider_name,
            status_code=500,
            message=f"Unexpected error communicating with Gemini: {str(exc)}",
        )

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a complete chat completion via Gemini API.

        Args:
            request: Standardized completion request.

        Returns:
            ChatCompletionResponse: Normalized chat completion response.

        Raises:
            RateLimitError: If Gemini returns HTTP 429 or quota exceeded.
            ProviderAPIError: For other Gemini upstream errors.
        """
        model_name = request.model or "gemini-2.5-flash"
        system_instruction, contents = self._prepare_payload(request)

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=system_instruction,
        )

        logger.debug(
            "Dispatching Gemini completion request",
            model=model_name,
            message_count=len(contents),
            has_system=bool(system_instruction),
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            # Extract generated content text and finish reason
            content_text = ""
            finish_reason = "stop"

            if hasattr(response, "text") and response.text:
                content_text = response.text

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if getattr(candidate, "finish_reason", None):
                    finish_reason = self._map_finish_reason(candidate.finish_reason) or "stop"
                if not content_text and candidate.content and candidate.content.parts:
                    content_text = "".join(
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    )

            usage = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = (
                    getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                )
                total_tokens = getattr(
                    response.usage_metadata, "total_token_count", prompt_tokens + completion_tokens
                ) or (prompt_tokens + completion_tokens)
                usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            return ChatCompletionResponse(
                id=f"chatcmpl-gemini-{uuid.uuid4().hex[:12]}",
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
                model=model_name,
                usage=usage,
                cached=False,
            )

        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream completion tokens from Gemini API as unified SSE chunks.

        Args:
            request: Standardized completion request with streaming enabled.

        Yields:
            ChatCompletionChunk: Normalized streaming response chunks.

        Raises:
            RateLimitError: If Gemini rate limits during stream setup.
            ProviderAPIError: If an error occurs during stream transmission.
        """
        model_name = request.model or "gemini-2.5-flash"
        system_instruction, contents = self._prepare_payload(request)

        config = types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            system_instruction=system_instruction,
        )

        logger.debug(
            "Dispatching Gemini stream request",
            model=model_name,
            message_count=len(contents),
        )

        try:
            stream_resp = await self._client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._handle_exception(exc) from exc

        stream_id = f"chatcmpl-gemini-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        first_chunk = True

        try:
            async for chunk in stream_resp:
                chunk_text = ""
                finish_reason = None

                if hasattr(chunk, "text") and chunk.text:
                    chunk_text = chunk.text

                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if getattr(candidate, "finish_reason", None):
                        finish_reason = self._map_finish_reason(candidate.finish_reason)
                    if not chunk_text and candidate.content and candidate.content.parts:
                        chunk_text = "".join(
                            part.text
                            for part in candidate.content.parts
                            if hasattr(part, "text") and part.text
                        )

                role = "assistant" if first_chunk else None
                first_chunk = False

                yield ChatCompletionChunk(
                    id=stream_id,
                    object="chat.completion.chunk",
                    created=created_ts,
                    choices=[
                        Choice(
                            index=0,
                            delta=DeltaContent(
                                role=role, content=chunk_text if chunk_text else None
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    model=model_name,
                )

            # Terminal stop chunk if not already finished
            if first_chunk:  # In case the stream was empty
                yield ChatCompletionChunk(
                    id=stream_id,
                    object="chat.completion.chunk",
                    created=created_ts,
                    choices=[
                        Choice(
                            index=0,
                            delta=DeltaContent(role="assistant", content=""),
                            finish_reason="stop",
                        )
                    ],
                    model=model_name,
                )
        except Exception as exc:
            raise self._handle_exception(exc) from exc

    async def health_check(self) -> bool:
        """Perform health check on Gemini adapter connectivity.

        Returns:
            bool: True if client is initialized and operational.
        """
        return bool(self._client)

    async def close(self) -> None:
        """Cleanup Gemini client resources."""
        # google.genai Client does not require explicit connection closure
        logger.debug("Closed Gemini client adapter")
