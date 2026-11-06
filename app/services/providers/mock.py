"""Fully functional mock LLM provider implementation for local testing and development.

Provides deterministic and realistic canned responses, token estimation, and simulated
SSE streaming with realistic word-by-word chunking and latency.
"""

import asyncio
import re
import time
import uuid
from collections.abc import AsyncIterator

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
from app.services.providers.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class MockProvider(BaseLLMProvider):
    """Mock LLM provider generating realistic responses without upstream network calls.

    Matches user queries against keyword categories (e.g., password reset, coding,
    general knowledge, database, security) to provide contextually relevant answers.
    Supports both complete batch generation and token-by-token simulated streaming.
    """

    provider_name: str = "mock"

    _CANNED_RESPONSES: list[tuple[tuple[str, ...], str]] = [
        (
            ("password", "reset", "forgot password", "unlock", "credentials"),
            (
                "To reset your account password, please follow these steps:\n"
                "1. Navigate to the login portal and click 'Forgot Password'.\n"
                "2. Enter your registered corporate email address.\n"
                "3. Check your inbox for a secure one-time password (OTP) or password reset link.\n"
                "4. Follow the link and choose a strong password containing at least 12 characters, "
                "including uppercase, lowercase, numbers, and symbols.\n"
                "5. Save changes and log in with your new credentials. If you do not receive the email, "
                "contact IT support at support@company.internal."
            ),
        ),
        (
            (
                "python",
                "fastapi",
                "code",
                "async",
                "function",
                "algorithm",
                "debug",
                "programming",
                "script",
            ),
            (
                "Here is a clean, idiomatic Python example using FastAPI and asyncio:\n\n"
                "```python\n"
                "import asyncio\n"
                "from fastapi import FastAPI, HTTPException\n\n"
                "app = FastAPI(title='Data Pipeline Service')\n\n"
                "@app.get('/items/{item_id}')\n"
                "async def read_item(item_id: int):\n"
                "    if item_id <= 0:\n"
                "        raise HTTPException(status_code=400, detail='Invalid item ID')\n"
                "    await asyncio.sleep(0.01)  # Simulate non-blocking async I/O\n"
                "    return {'item_id': item_id, 'status': 'processed', 'success': True}\n"
                "```\n\n"
                "Key best practices applied:\n"
                "- Non-blocking async endpoints for high I/O throughput\n"
                "- Explicit HTTP exception raising with clear status codes and details\n"
                "- Type annotations and Pydantic validation integration."
            ),
        ),
        (
            ("sql", "database", "postgres", "redis", "vector index", "vector search"),
            (
                "Optimizing database and vector cache performance involves several core strategies:\n\n"
                "1. **Vector Indexing (HNSW / Flat)**: When using Redis Vector Search (RediSearch), "
                "HNSW provides logarithmic search time with high recall at scale.\n"
                "2. **Connection Pooling**: Always reuse connection pools (e.g. `redis.asyncio.ConnectionPool`) "
                "to minimize handshake latency.\n"
                "3. **Semantic Caching**: Store query vector embeddings alongside pre-computed responses "
                "with an appropriate cosine distance threshold (e.g. 0.08).\n"
                "4. **TTL Policy**: Set automatic expiration on keys to prevent unbounded memory growth."
            ),
        ),
        (
            ("pii", "redact", "mask", "sanitize", "security", "gdpr", "compliance"),
            (
                "Data Privacy & PII Handling Policy:\n"
                "- Sensitive elements (SSN, credit card numbers, email addresses, phone numbers, and API keys) "
                "must be detected and redacted prior to persistent storage or upstream forwarding.\n"
                "- Redaction tokens follow standard format: `[REDACTED_SSN]`, `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`.\n"
                "- Strict zero-retention policies apply to unmasked payload context."
            ),
        ),
        (
            ("gateway", "proxy", "circuit breaker", "fallback", "deepeval", "eval"),
            (
                "The LLM Evaluation Gateway provides high-throughput resilience and quality assurance:\n\n"
                "- **Dynamic Fallback**: Automatically cascades across providers (e.g., Gemini -> Anthropic -> OpenAI) "
                "when errors or rate limits (HTTP 429) are detected.\n"
                "- **Circuit Breakers**: Enforces finite failure windows to isolate degraded upstream endpoints.\n"
                "- **Evaluation Proxy**: Evaluates response relevancy, hallucination rate, and bias using DeepEval metrics."
            ),
        ),
        (
            ("weather", "capital", "history", "who", "what", "science", "explain", "physics"),
            (
                "Here is an overview based on established scientific and general knowledge:\n\n"
                "1. **Core Principle**: Natural phenomena and scientific theories are modeled through empirical observation, "
                "mathematical formulations, and reproducible experimentation.\n"
                "2. **Key Concepts**: Energy conservation, entropy, and thermodynamic equilibrium govern physical systems.\n"
                "3. **Conclusion**: Understanding foundational mechanisms allows for accurate forecasting and system design."
            ),
        ),
    ]

    _DEFAULT_RESPONSE: str = (
        "Thank you for your inquiry. As an AI assistant proxy, I have processed your request. "
        "All systems are operating normally, and responses are formatted according to OpenAI-compatible specifications."
    )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on a 4-character per token heuristic.

        Args:
            text: Input string to estimate.

        Returns:
            int: Estimated token count (minimum 1 token for non-empty text).
        """
        if not text:
            return 0
        return max(1, (len(text) + 3) // 4)

    def _matches_keyword(self, keyword: str, text: str) -> bool:
        """Check if keyword matches text using word boundary matching.

        Args:
            keyword: Single word or multi-word phrase to match.
            text: Full lowercased text string.

        Returns:
            bool: True if matched, False otherwise.
        """
        if " " in keyword:
            return keyword in text
        pattern = rf"\b{re.escape(keyword)}\b"
        return bool(re.search(pattern, text))

    def _select_response(self, request: ChatCompletionRequest) -> str:
        """Select a canned response based on message content keywords.

        Args:
            request: The completion request.

        Returns:
            str: The selected response content.
        """
        # Combine all message contents into a single searchable text block
        full_text = " ".join(msg.content.lower() for msg in request.messages)

        for keywords, response in self._CANNED_RESPONSES:
            if any(self._matches_keyword(kw, full_text) for kw in keywords):
                return response

        return self._DEFAULT_RESPONSE

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a canned mock response with accurate token estimation.

        Args:
            request: Standardized chat completion request.

        Returns:
            ChatCompletionResponse: Populated completion response with usage stats.
        """
        response_text = self._select_response(request)
        model_name = request.model or "mock-gpt-4o"
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"

        prompt_tokens = sum(self._estimate_tokens(m.content) for m in request.messages)
        completion_tokens = self._estimate_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        logger.debug(
            "Generated mock completion",
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            id=completion_id,
        )

        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            model=model_name,
            usage=usage,
            cached=False,
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream the canned response token-by-token with realistic latency.

        Args:
            request: Standardized chat completion request with streaming enabled.

        Yields:
            ChatCompletionChunk: Incremental token chunks matching OpenAI SSE schema.
        """
        response_text = self._select_response(request)
        model_name = request.model or "mock-gpt-4o"
        stream_id = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        # Split response into words for realistic streaming granularity
        words = response_text.split(" ")
        num_words = len(words)

        logger.debug(
            "Starting mock stream generation",
            model=model_name,
            word_count=num_words,
            id=stream_id,
        )

        for idx, word in enumerate(words):
            # Append space to all but the last word
            content_piece = word if idx == num_words - 1 else f"{word} "

            # First chunk introduces role="assistant"
            role = "assistant" if idx == 0 else None

            chunk = ChatCompletionChunk(
                id=stream_id,
                object="chat.completion.chunk",
                created=created_ts,
                choices=[
                    Choice(
                        index=0,
                        delta=DeltaContent(role=role, content=content_piece),
                        finish_reason=None,
                    )
                ],
                model=model_name,
            )
            yield chunk

            # Simulate network latency between token generation
            await asyncio.sleep(0.05)

        # Final terminal chunk with finish_reason="stop"
        final_chunk = ChatCompletionChunk(
            id=stream_id,
            object="chat.completion.chunk",
            created=created_ts,
            choices=[
                Choice(
                    index=0,
                    delta=DeltaContent(role=None, content=None),
                    finish_reason="stop",
                )
            ],
            model=model_name,
        )
        yield final_chunk

    async def health_check(self) -> bool:
        """Verify mock provider readiness.

        Returns:
            bool: Always True for mock provider.
        """
        return True

    async def close(self) -> None:
        """Cleanup mock provider resources.

        Returns:
            None
        """
        return None
