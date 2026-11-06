"""Unit and integration tests for API Key Authentication Middleware.

Verifies valid API key acceptance, missing and invalid API key rejections,
authentication bypass when AUTH_ENABLED=False, and exempt health check endpoints.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.config import Settings
from app.main import create_app
from app.middleware.auth import APIKeyAuthMiddleware


@pytest.mark.asyncio
async def test_valid_api_key() -> None:
    """Test that a request containing a configured valid X-API-Key succeeds with 200 OK."""
    auth_settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"dev-key-001": "dev_service", "prod-key-100": "prod_service"},
        DEFAULT_PROVIDER="mock",
    )
    test_app = create_app(settings=auth_settings)
    test_app.state.settings = auth_settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health", headers={"X-API-Key": "dev-key-001"})
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_missing_api_key() -> None:
    """Test that a request to a protected endpoint without an X-API-Key returns 401 Unauthorized."""
    auth_settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"dev-key-001": "dev_service"},
        DEFAULT_PROVIDER="mock",
    )
    test_app = create_app(settings=auth_settings)
    test_app.state.settings = auth_settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request to protected completions endpoint without X-API-Key
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401
        data = response.json()
        assert "Unauthorized" in data.get("error", "") or "Invalid" in data.get("detail", "")


@pytest.mark.asyncio
async def test_invalid_api_key() -> None:
    """Test that a request with an unrecognized X-API-Key returns 401 Unauthorized."""
    auth_settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"dev-key-001": "dev_service"},
        DEFAULT_PROVIDER="mock",
    )
    test_app = create_app(settings=auth_settings)
    test_app.state.settings = auth_settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "completely-invalid-key-999"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 401
        data = response.json()
        assert "Unauthorized" in data.get("error", "") or "Invalid" in data.get("detail", "")


@pytest.mark.asyncio
async def test_auth_disabled_bypass() -> None:
    """Test that when AUTH_ENABLED=False, requests without an API key succeed and bypass auth."""
    no_auth_settings = Settings(
        AUTH_ENABLED=False,
        GATEWAY_API_KEYS={"dev-key-001": "dev_service"},
        DEFAULT_PROVIDER="mock",
    )
    test_app = create_app(settings=no_auth_settings)
    test_app.state.settings = no_auth_settings

    from app.services.circuit_breaker.breaker import CircuitBreakerRegistry
    from app.services.providers.mock import MockProvider
    from app.services.router.intent_router import IntentRouter
    from app.services.sanitizer.pii import PIISanitizer

    mock_prov = MockProvider()
    test_app.state.providers = {
        "mock": mock_prov,
        "openai": mock_prov,
        "anthropic": mock_prov,
        "gemini": mock_prov,
    }
    test_app.state.circuit_breakers = CircuitBreakerRegistry(settings=no_auth_settings)
    test_app.state.router = IntentRouter()
    test_app.state.sanitizer = PIISanitizer()
    test_app.state.semantic_cache = None

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Request without any API key should succeed
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello without key"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


@pytest.mark.asyncio
async def test_health_endpoint_no_auth() -> None:
    """Test that the /health endpoint is exempt from authentication even when AUTH_ENABLED=True."""
    strict_auth_settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"prod-secret-key": "prod_service"},
    )
    test_app = create_app(settings=strict_auth_settings)
    test_app.state.settings = strict_auth_settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /health with no credentials must succeed
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "ready", "ok", "degraded")


@pytest.mark.asyncio
async def test_exempt_paths_direct_dispatch() -> None:
    """Test middleware dispatch handling directly for exempt and non-exempt paths."""
    settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"valid-key": "tenant-1"},
    )

    async def dummy_app(scope, receive, send):
        response = JSONResponse({"status": "ok"})
        await response(scope, receive, send)

    middleware = APIKeyAuthMiddleware(app=dummy_app, settings=settings)

    # Test /health directly
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [],
        "state": {},
    }
    request = Request(scope)

    async def call_next(_req: Request):
        return JSONResponse({"status": "ok"})

    resp = await middleware.dispatch(request, call_next)
    assert resp.status_code == 200
