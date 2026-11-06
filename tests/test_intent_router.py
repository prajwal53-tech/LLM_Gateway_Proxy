"""Unit tests for the IntentRouter dynamic routing service.

Verifies heuristic routing for programming code patterns, SQL queries, long context payloads,
standard queries, custom routing strategy headers, and explicit model/provider overrides.
"""

import pytest

from app.api.v1.schemas import ChatCompletionRequest, ChatMessage
from app.services.router.intent_router import IntentRouter


@pytest.fixture
def router() -> IntentRouter:
    """Fixture providing an instance of IntentRouter."""
    return IntentRouter()


def test_route_code_prompt(router: IntentRouter) -> None:
    """Test that prompts containing code definitions route to Anthropic Claude."""
    request_def = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="def calculate_fibonacci(n: int) -> int: pass")],
    )
    provider, model = router.route(request_def)
    assert provider == "anthropic"
    assert "claude" in model

    request_class = ChatCompletionRequest(
        messages=[
            ChatMessage(role="user", content="class DataProcessor:\n    def __init__(self): pass")
        ],
    )
    provider, model = router.route(request_class)
    assert provider == "anthropic"
    assert "claude" in model


def test_route_sql_prompt(router: IntentRouter) -> None:
    """Test that prompts containing SQL keywords route to Anthropic Claude."""
    request_sql = ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content="SELECT user_id, email, created_at FROM users WHERE active = true;",
            )
        ],
    )
    provider, model = router.route(request_sql)
    assert provider == "anthropic"
    assert "claude" in model


def test_route_long_prompt(router: IntentRouter) -> None:
    """Test that prompts exceeding 2000 characters route to OpenAI GPT-4o."""
    long_content = "This is an extensive analysis context paragraph. " * 60
    assert len(long_content) > 2000

    request_long = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=long_content)],
    )
    provider, model = router.route(request_long)
    assert provider == "openai"
    assert "gpt-4o" in model


def test_route_standard_prompt(router: IntentRouter) -> None:
    """Test that short simple prompts default to cost-effective Gemini Flash."""
    request_standard = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is the weather like in Seattle today?")],
    )
    provider, model = router.route(request_standard)
    assert provider == "gemini"
    assert "gemini" in model


def test_route_quality_header(router: IntentRouter) -> None:
    """Test that X-Routing-Strategy: quality-first routes to Anthropic Claude."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="What is quantum computing?")],
    )
    headers = {"X-Routing-Strategy": "quality-first"}
    provider, model = router.route(request, headers=headers)
    assert provider == "anthropic"
    assert "claude" in model


def test_route_cost_header(router: IntentRouter) -> None:
    """Test that X-Routing-Strategy: cost-optimized routes to Google Gemini."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Translate this sentence to French.")],
    )
    headers = {"x-routing-strategy": "cost-optimized"}
    provider, model = router.route(request, headers=headers)
    assert provider == "gemini"
    assert "gemini" in model


def test_route_explicit_model(router: IntentRouter) -> None:
    """Test that specifying request.model directly overrides heuristics and routes to the model's provider."""
    # Explicit OpenAI model
    req_openai = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Simple prompt")],
    )
    provider, model = router.route(req_openai)
    assert provider == "openai"
    assert model == "gpt-4o"

    # Explicit Claude model
    req_claude = ChatCompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[ChatMessage(role="user", content="Simple prompt")],
    )
    provider, model = router.route(req_claude)
    assert provider == "anthropic"
    assert model == "claude-3-5-sonnet-20241022"

    # Explicit Gemini model
    req_gemini = ChatCompletionRequest(
        model="gemini-1.5-flash",
        messages=[ChatMessage(role="user", content="def foo(): pass")],
    )
    provider, model = router.route(req_gemini)
    assert provider == "gemini"
    assert model == "gemini-1.5-flash"

    # Explicit Mock model
    req_mock = ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[ChatMessage(role="user", content="def foo(): pass")],
    )
    provider, model = router.route(req_mock)
    assert provider == "mock"
    assert model == "mock-gpt-4o"
