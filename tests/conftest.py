"""Shared pytest fixtures and test configuration for llm-eval-proxy.

Provides reusable fixtures for application settings, mock LLM providers,
PII sanitizers, sample request payloads, and asynchronous FastAPI HTTP test clients.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.schemas import ChatCompletionRequest, ChatMessage
from app.core.config import Settings
from app.main import app
from app.services.circuit_breaker.breaker import CircuitBreakerRegistry
from app.services.providers.mock import MockProvider
from app.services.router.intent_router import IntentRouter
from app.services.sanitizer.pii import PIISanitizer


@pytest.fixture
def settings() -> Settings:
    """Fixture providing isolated application configuration for testing.

    Configures the proxy with auth disabled by default for simplified unit testing,
    mock LLM provider as default, and localhost Redis endpoint.

    Returns:
        Settings: Initialized test settings instance.
    """
    return Settings(
        APP_NAME="llm-eval-proxy-test",
        APP_VERSION="0.1.0-test",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        AUTH_ENABLED=False,
        GATEWAY_API_KEYS={"dev-key-001": "dev_service", "test-key-999": "test_tenant"},
        REDIS_URL="redis://localhost:6379",
        CACHE_TTL_SECONDS=3600,
        CACHE_SIMILARITY_THRESHOLD=0.08,
        EMBEDDING_PROVIDER="local",
        EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2",
        DEFAULT_PROVIDER="mock",
        CB_FAILURE_THRESHOLD=5,
        CB_FAILURE_WINDOW_SECONDS=30,
        CB_COOLDOWN_SECONDS=60,
        CB_HALF_OPEN_PROBE_RATIO=0.5,
        CB_REQUEST_TIMEOUT_MS=5000,
        FALLBACK_CHAIN=["mock", "gemini", "anthropic", "openai"],
    )


@pytest.fixture
def mock_provider() -> MockProvider:
    """Fixture providing an instance of MockProvider for testing.

    Returns:
        MockProvider: Fully functional deterministic mock provider.
    """
    return MockProvider()


@pytest.fixture
def pii_sanitizer() -> PIISanitizer:
    """Fixture providing an instance of PIISanitizer with precompiled patterns.

    Returns:
        PIISanitizer: Regex-based PII detector and sanitizer.
    """
    return PIISanitizer()


@pytest.fixture
def sample_request() -> ChatCompletionRequest:
    """Fixture providing a standard non-streaming chat completion request.

    Returns:
        ChatCompletionRequest: Valid OpenAI-compatible request payload.
    """
    return ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[
            ChatMessage(role="user", content="Hello, tell me about password reset policies.")
        ],
        temperature=0.7,
        max_tokens=256,
        stream=False,
    )


@pytest.fixture
def sample_stream_request() -> ChatCompletionRequest:
    """Fixture providing a streaming chat completion request.

    Returns:
        ChatCompletionRequest: Valid streaming chat completion request payload.
    """
    return ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[ChatMessage(role="user", content="Explain python async functions and FastAPI.")],
        temperature=0.7,
        max_tokens=256,
        stream=True,
    )


@pytest.fixture
async def app_client(settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client fixture configured for testing the FastAPI application.

    Initializes application state with test settings and mock components to avoid
    external Redis or upstream network dependencies during unit and endpoint testing.

    Yields:
        AsyncClient: Configured httpx AsyncClient with ASGITransport and default headers.
    """
    # Ensure app state is fully populated with test fixtures
    mock_prov = MockProvider()
    app.state.settings = settings
    app.state.providers = {
        "mock": mock_prov,
        "openai": mock_prov,
        "anthropic": mock_prov,
        "gemini": mock_prov,
    }
    app.state.circuit_breakers = CircuitBreakerRegistry(settings=settings)
    app.state.router = IntentRouter()
    app.state.sanitizer = PIISanitizer()
    app.state.semantic_cache = None
    app.state.vector_store = None
    app.state.embedding_service = None

    transport = ASGITransport(app=app)
    headers = {
        "X-API-Key": "dev-key-001",
        "Content-Type": "application/json",
    }

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers=headers,
    ) as client:
        yield client
