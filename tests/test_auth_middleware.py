"""Unit tests for APIKeyAuthMiddleware."""

from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from app.core.config import Settings
from app.middleware.auth import APIKeyAuthMiddleware


def create_test_app(settings: Settings) -> FastAPI:
    """Create a FastAPI test app wrapped with APIKeyAuthMiddleware."""
    app = FastAPI()
    app.add_middleware(APIKeyAuthMiddleware, settings=settings)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics():
        return "metrics_data"

    @app.get("/api/v1/chat")
    async def chat(request: Request):
        return {"client_id": getattr(request.state, "client_id", None)}

    return app


def test_exempt_endpoints_bypass_auth():
    """Test that exempt endpoints do not require API keys even if auth is enabled."""
    settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"secret-123": "tenant_alpha"},
    )
    app = create_test_app(settings)
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

    # Readiness check
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}

    # Metrics check
    res = client.get("/metrics")
    assert res.status_code == 200


def test_auth_disabled_bypass():
    """Test that all endpoints succeed with 'dev_bypass' when AUTH_ENABLED is False."""
    settings = Settings(
        AUTH_ENABLED=False,
        GATEWAY_API_KEYS={"secret-123": "tenant_alpha"},
    )
    app = create_test_app(settings)
    client = TestClient(app)

    res = client.get("/api/v1/chat")
    assert res.status_code == 200
    assert res.json() == {"client_id": "dev_bypass"}


def test_missing_api_key_returns_401():
    """Test that missing X-API-Key header returns 401 Unauthorized."""
    settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"secret-123": "tenant_alpha"},
    )
    app = create_test_app(settings)
    client = TestClient(app)

    res = client.get("/api/v1/chat")
    assert res.status_code == 401
    data = res.json()
    assert data["error"] == "Unauthorized"
    assert "Invalid or missing API key" in data["detail"]


def test_invalid_api_key_returns_401():
    """Test that invalid X-API-Key header returns 401 Unauthorized."""
    settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"secret-123": "tenant_alpha"},
    )
    app = create_test_app(settings)
    client = TestClient(app)

    res = client.get("/api/v1/chat", headers={"X-API-Key": "invalid-key"})
    assert res.status_code == 401
    data = res.json()
    assert data["error"] == "Unauthorized"


def test_valid_api_key_sets_client_id():
    """Test that valid X-API-Key sets client_id on request.state and returns 200."""
    settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"secret-123": "tenant_alpha", "beta-456": "tenant_beta"},
    )
    app = create_test_app(settings)
    client = TestClient(app)

    res = client.get("/api/v1/chat", headers={"X-API-Key": "secret-123"})
    assert res.status_code == 200
    assert res.json() == {"client_id": "tenant_alpha"}

    res2 = client.get("/api/v1/chat", headers={"x-api-key": "beta-456"})
    assert res2.status_code == 200
    assert res2.json() == {"client_id": "tenant_beta"}
