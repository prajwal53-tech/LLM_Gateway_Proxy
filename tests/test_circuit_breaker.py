"""Unit tests for the CircuitBreaker and CircuitBreakerRegistry resilience mechanisms.

Verifies state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED), threshold tripping,
cooldown expiration, probe throttling, and exception handling.
"""

import time

import pytest

from app.core.config import Settings
from app.core.exceptions import CircuitOpenError, ProviderAPIError, RateLimitError
from app.services.circuit_breaker.breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)


@pytest.fixture
def breaker_settings() -> Settings:
    """Fixture providing test settings with short failure thresholds for testing."""
    return Settings(
        CB_FAILURE_THRESHOLD=5,
        CB_FAILURE_WINDOW_SECONDS=30,
        CB_COOLDOWN_SECONDS=1,  # 1 second cooldown for fast testing
        CB_HALF_OPEN_PROBE_RATIO=1.0,  # 100% probe ratio for deterministic testing
        CB_REQUEST_TIMEOUT_MS=1000,
    )


@pytest.mark.asyncio
async def test_initial_state_closed(breaker_settings: Settings) -> None:
    """Test that a newly initialized CircuitBreaker starts in the CLOSED state."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.state_value == 0
    assert await breaker.can_execute() is True


@pytest.mark.asyncio
async def test_successful_execution(breaker_settings: Settings) -> None:
    """Test that an asynchronous operation executes successfully through the circuit breaker."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)

    async def successful_call(val: int) -> int:
        return val * 2

    result = await breaker.execute(successful_call, 21)
    assert result == 42
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_failure_trips_circuit(breaker_settings: Settings) -> None:
    """Test that 5 consecutive provider failures trip the circuit breaker into OPEN state."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)

    async def failing_call() -> None:
        raise ProviderAPIError(provider_name="test-provider", message="Service Unavailable")

    for _ in range(4):
        with pytest.raises(ProviderAPIError):
            await breaker.execute(failing_call)
        assert breaker.state == CircuitState.CLOSED

    # 5th failure reaches threshold and trips circuit to OPEN
    with pytest.raises(ProviderAPIError):
        await breaker.execute(failing_call)

    assert breaker.state == CircuitState.OPEN
    assert breaker.state_value == 1


@pytest.mark.asyncio
async def test_open_circuit_raises(breaker_settings: Settings) -> None:
    """Test that attempting execution on an OPEN circuit immediately raises CircuitOpenError."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)
    breaker.state = CircuitState.OPEN
    breaker.last_failure_time = time.time()  # Recent failure, cooldown not elapsed

    async def dummy_call() -> str:
        return "success"

    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.execute(dummy_call)

    assert exc_info.value.provider_name == "test-provider"
    assert "OPEN" in exc_info.value.message


@pytest.mark.asyncio
async def test_cooldown_transitions_to_half_open(breaker_settings: Settings) -> None:
    """Test that after cooldown period elapses, the circuit transitions to HALF_OPEN."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)
    breaker.state = CircuitState.OPEN
    # Set last failure time far in the past to simulate elapsed cooldown
    breaker.last_failure_time = time.time() - (breaker_settings.CB_COOLDOWN_SECONDS + 5)

    can_exec = await breaker.can_execute()
    assert can_exec is True
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.state_value == 2


@pytest.mark.asyncio
async def test_success_in_half_open_closes(breaker_settings: Settings) -> None:
    """Test that a successful probe execution in HALF_OPEN state resets circuit back to CLOSED."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)
    breaker.state = CircuitState.HALF_OPEN

    async def successful_call() -> str:
        return "recovered"

    result = await breaker.execute(successful_call)
    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.state_value == 0
    assert len(breaker.failure_timestamps) == 0


@pytest.mark.asyncio
async def test_failure_in_half_open_opens(breaker_settings: Settings) -> None:
    """Test that a failed probe execution in HALF_OPEN state immediately transitions back to OPEN."""
    breaker = CircuitBreaker(provider_name="test-provider", settings=breaker_settings)
    breaker.state = CircuitState.HALF_OPEN

    async def failing_call() -> None:
        raise RateLimitError(provider_name="test-provider", message="Rate limit exceeded")

    with pytest.raises(RateLimitError):
        await breaker.execute(failing_call)

    assert breaker.state == CircuitState.OPEN
    assert breaker.state_value == 1


def test_registry_creates_breakers(breaker_settings: Settings) -> None:
    """Test that CircuitBreakerRegistry creates, caches, and retrieves breaker instances per provider."""
    registry = CircuitBreakerRegistry(settings=breaker_settings)

    openai_breaker = registry.get("openai")
    assert openai_breaker.provider_name == "openai"
    assert openai_breaker.state == CircuitState.CLOSED

    # Retrieving same provider returns the cached instance
    assert registry.get("openai") is openai_breaker

    anthropic_breaker = registry.get("anthropic")
    assert anthropic_breaker.provider_name == "anthropic"
    assert anthropic_breaker is not openai_breaker

    all_breakers = registry.get_all()
    assert len(all_breakers) == 2
    assert "openai" in all_breakers
    assert "anthropic" in all_breakers
