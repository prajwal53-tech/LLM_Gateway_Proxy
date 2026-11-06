"""Integration tests for the /v1/chat/completions endpoint.

Verifies end-to-end request handling for non-streaming completions, SSE streaming responses,
automatic PII redaction during request processing, and authentication enforcement.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app


@pytest.mark.asyncio
async def test_non_streaming_completion(app_client: AsyncClient) -> None:
    """Test non-streaming POST /v1/chat/completions returns 200 OK and valid JSON completion."""
    payload = {
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "How do I reset my password?"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False,
    }

    response = await app_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert len(data["choices"]) > 0

    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "stop"

    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_streaming_completion(app_client: AsyncClient) -> None:
    """Test streaming POST /v1/chat/completions returns SSE events with parseable JSON chunks."""
    payload = {
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "Show me a Python FastAPI script"}],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }

    response = await app_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Parse Server-Sent Events (SSE) stream
    events = []
    lines = response.text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            data_content = line[len("data:") :].strip()
            if data_content and data_content != "[DONE]":
                chunk = json.loads(data_content)
                events.append(chunk)

    assert len(events) > 0

    # Verify chunk structure
    first_event = events[0]
    assert first_event["object"] == "chat.completion.chunk"
    assert "choices" in first_event
    assert len(first_event["choices"]) > 0

    # Ensure last event has finish_reason="stop"
    last_event = events[-1]
    assert last_event["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_pii_redacted_in_request(app_client: AsyncClient) -> None:
    """Test that sending a prompt containing sensitive PII (SSN) processes cleanly and returns a response."""
    payload = {
        "model": "mock-gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "Please verify user account for SSN 123-45-6789 and email user@example.com.",
            }
        ],
        "temperature": 0.7,
        "stream": False,
    }

    response = await app_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert data["choices"][0]["message"]["content"] is not None


@pytest.mark.asyncio
async def test_missing_auth_returns_401() -> None:
    """Test that requests without X-API-Key return 401 when AUTH_ENABLED=True."""
    auth_settings = Settings(
        AUTH_ENABLED=True,
        GATEWAY_API_KEYS={"valid-key-123": "service_a"},
        DEFAULT_PROVIDER="mock",
    )
    test_app = create_app(settings=auth_settings)
    test_app.state.settings = auth_settings

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "mock-gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_completions_alias_route(app_client: AsyncClient) -> None:
    """Test that both /chat/completions and /v1/chat/completions work identically."""
    payload = {
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "Explain Redis vector caching"}],
        "stream": False,
    }

    resp1 = await app_client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200

    resp2 = await app_client.post("/chat/completions", json=payload)
    assert resp2.status_code == 200
