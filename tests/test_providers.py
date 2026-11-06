"""Unit tests for MockProvider and ProviderFactory implementations.

Verifies deterministic non-streaming and streaming generation, finish reason termination,
canned response selection, token estimation, and factory lifecycle management.
"""

import pytest

from app.api.v1.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from app.core.config import Settings
from app.services.providers.base import BaseLLMProvider
from app.services.providers.factory import ProviderFactory
from app.services.providers.mock import MockProvider


@pytest.mark.asyncio
async def test_mock_generate(
    mock_provider: MockProvider, sample_request: ChatCompletionRequest
) -> None:
    """Test non-streaming generation returns a valid ChatCompletionResponse."""
    response = await mock_provider.generate(sample_request)

    assert isinstance(response, ChatCompletionResponse)
    assert response.id.startswith("chatcmpl-mock-")
    assert response.object == "chat.completion"
    assert response.model == sample_request.model
    assert response.cached is False
    assert len(response.choices) == 1

    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.message is not None
    assert choice.message.role == "assistant"
    assert len(choice.message.content) > 0

    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert (
        response.usage.total_tokens
        == response.usage.prompt_tokens + response.usage.completion_tokens
    )


@pytest.mark.asyncio
async def test_mock_stream(
    mock_provider: MockProvider, sample_stream_request: ChatCompletionRequest
) -> None:
    """Test streaming generation yields multiple ChatCompletionChunk objects with content."""
    chunks: list[ChatCompletionChunk] = []

    async for chunk in mock_provider.stream(sample_stream_request):
        assert isinstance(chunk, ChatCompletionChunk)
        assert chunk.object == "chat.completion.chunk"
        assert chunk.model == sample_stream_request.model
        assert len(chunk.choices) == 1
        chunks.append(chunk)

    # Must have yielded multiple incremental chunks
    assert len(chunks) > 1

    # Verify that the first content chunk defines role="assistant"
    first_chunk = chunks[0]
    assert first_chunk.choices[0].delta is not None
    assert first_chunk.choices[0].delta.role == "assistant"

    # Concatenate all streamed text
    streamed_text = "".join(
        c.choices[0].delta.content
        for c in chunks
        if c.choices[0].delta and c.choices[0].delta.content
    )
    assert len(streamed_text) > 0
    assert "FastAPI" in streamed_text or "async" in streamed_text or "Python" in streamed_text


@pytest.mark.asyncio
async def test_mock_stream_has_finish_reason(
    mock_provider: MockProvider, sample_stream_request: ChatCompletionRequest
) -> None:
    """Test that the terminal streaming chunk has finish_reason='stop'."""
    chunks: list[ChatCompletionChunk] = []

    async for chunk in mock_provider.stream(sample_stream_request):
        chunks.append(chunk)

    assert len(chunks) > 0
    last_chunk = chunks[-1]
    assert last_chunk.choices[0].finish_reason == "stop"
    assert last_chunk.choices[0].delta.content is None


def test_provider_factory_creates_mock(settings: Settings) -> None:
    """Test that ProviderFactory.create('mock', settings) returns an instance of MockProvider."""
    provider = ProviderFactory.create("mock", settings)
    assert isinstance(provider, MockProvider)
    assert isinstance(provider, BaseLLMProvider)
    assert provider.provider_name == "mock"


def test_provider_factory_unknown_raises(settings: Settings) -> None:
    """Test that ProviderFactory.create with an unsupported provider raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        ProviderFactory.create("unknown-provider-xyz", settings)
    assert "Unsupported provider" in str(exc_info.value)


def test_provider_factory_list_supported() -> None:
    """Test that ProviderFactory.list_supported_providers lists all configured providers."""
    supported = ProviderFactory.list_supported_providers()
    assert "mock" in supported
    assert "openai" in supported
    assert "anthropic" in supported
    assert "gemini" in supported


def test_provider_factory_create_all(settings: Settings) -> None:
    """Test that ProviderFactory.create_all instantiates all registered adapters."""
    providers = ProviderFactory.create_all(settings)
    assert "mock" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "gemini" in providers


@pytest.mark.asyncio
async def test_mock_canned_responses(mock_provider: MockProvider) -> None:
    """Test that MockProvider selects matching canned responses based on keyword patterns."""
    # Test coding query
    code_req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Show me python async code")],
    )
    resp = await mock_provider.generate(code_req)
    assert (
        "FastAPI" in resp.choices[0].message.content or "asyncio" in resp.choices[0].message.content
    )

    # Test database/SQL query
    sql_req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="How do I optimize redis database query performance?")
        ],
    )
    resp = await mock_provider.generate(sql_req)
    assert "Redis" in resp.choices[0].message.content or "Vector" in resp.choices[0].message.content

    # Test PII query
    pii_req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is the pii redact policy?")],
    )
    resp = await mock_provider.generate(pii_req)
    assert "REDACTED" in resp.choices[0].message.content or "PII" in resp.choices[0].message.content


@pytest.mark.asyncio
async def test_mock_provider_health_check_and_close(mock_provider: MockProvider) -> None:
    """Test health check and resource close lifecycle on MockProvider."""
    assert await mock_provider.health_check() is True
    await mock_provider.close()
