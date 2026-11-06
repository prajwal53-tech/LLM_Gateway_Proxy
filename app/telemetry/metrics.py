"""Prometheus Metrics for LLM Gateway and Evaluation Proxy.

Defines standard metric instruments for tracking request throughput, latency,
time-to-first-token (TTFT), token consumption, circuit breaker state, semantic cache
operations, and PII detections.
"""

from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _FallbackMetric:
        """Fallback metric instrument when prometheus_client is not installed."""

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: list[str] | tuple[str, ...] = (),
            **_kwargs: Any,
        ) -> None:
            self._name = name
            self._doc = documentation
            self._labelnames = labelnames

        def labels(self, *_args: Any, **_kwargs: Any) -> "_FallbackMetric":
            return self

        def inc(self, _amount: float = 1.0) -> None:
            pass

        def dec(self, _amount: float = 1.0) -> None:
            pass

        def set(self, _value: float) -> None:
            pass

        def observe(self, _amount: float) -> None:
            pass

    Counter = _FallbackMetric  # type: ignore
    Gauge = _FallbackMetric  # type: ignore
    Histogram = _FallbackMetric  # type: ignore

    def generate_latest() -> bytes:  # type: ignore
        return b"# Prometheus metrics fallback\n"

# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "llm_gateway_requests_total",
    "Total requests processed",
    ["provider", "model", "status", "cache_hit", "client_id"],
)

TTFT_SECONDS = Histogram(
    "llm_gateway_ttft_seconds",
    "Time to first token in seconds",
    ["provider", "model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

TOKENS_TOTAL = Counter(
    "llm_gateway_tokens_consumed_total",
    "Total tokens consumed",
    ["provider", "model", "token_type"],
)

CIRCUIT_STATE = Gauge(
    "llm_gateway_circuit_breaker_state",
    "Circuit breaker state per provider (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["provider"],
)

CACHE_OPERATIONS = Counter(
    "llm_gateway_cache_operations_total",
    "Cache operations",
    ["operation", "result"],
)

PII_DETECTIONS = Counter(
    "llm_gateway_pii_detections_total",
    "PII detections",
    ["pii_type"],
)

ACTIVE_REQUESTS = Gauge(
    "llm_gateway_active_requests",
    "Currently active requests",
    ["provider"],
)

REQUEST_DURATION = Histogram(
    "llm_gateway_request_duration_seconds",
    "Total request duration",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def record_request(
    provider: str,
    model: str,
    status: str,
    cache_hit: bool | str,
    client_id: str = "unknown",
) -> None:
    """Record an LLM request completion in the Prometheus requests counter.

    Args:
        provider: Identifier of the LLM provider (e.g. 'openai', 'gemini', 'mock').
        model: Model identifier (e.g. 'gpt-4o', 'gemini-1.5-pro').
        status: Status code or outcome description (e.g. '200', '500', 'success', 'error').
        cache_hit: Whether the response was served from semantic cache (boolean or 'true'/'false').
        client_id: Authenticated client/tenant identifier.
    """
    cache_hit_str = (
        str(cache_hit).lower() if isinstance(cache_hit, bool) else str(cache_hit).lower()
    )
    REQUESTS_TOTAL.labels(
        provider=provider,
        model=model,
        status=str(status),
        cache_hit=cache_hit_str,
        client_id=client_id or "unknown",
    ).inc()


def record_ttft(
    provider: str,
    model: str,
    duration: float,
) -> None:
    """Record the time to first token (TTFT) for a streaming response.

    Args:
        provider: Identifier of the LLM provider.
        model: Model identifier.
        duration: Latency in seconds from request start to first streamed chunk.
    """
    TTFT_SECONDS.labels(
        provider=provider,
        model=model,
    ).observe(duration)


def record_tokens(
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record prompt and completion token counts.

    Args:
        provider: Identifier of the LLM provider.
        model: Model identifier.
        prompt_tokens: Number of tokens in the prompt payload.
        completion_tokens: Number of tokens generated in the completion payload.
    """
    if prompt_tokens > 0:
        TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            token_type="prompt",
        ).inc(prompt_tokens)

    if completion_tokens > 0:
        TOKENS_TOTAL.labels(
            provider=provider,
            model=model,
            token_type="completion",
        ).inc(completion_tokens)


def update_circuit_state(
    provider: str,
    state_value: float | int,
) -> None:
    """Set the circuit breaker state gauge for a provider.

    Args:
        provider: Identifier of the LLM provider.
        state_value: Numeric representation of circuit state (e.g. 0=CLOSED, 1=HALF_OPEN, 2=OPEN).
    """
    CIRCUIT_STATE.labels(
        provider=provider,
    ).set(state_value)


def record_cache_operation(
    operation: str,
    result: str,
) -> None:
    """Record a semantic cache operation outcome (e.g. hit, miss, write, error).

    Args:
        operation: Operation type ('lookup', 'store', 'delete', 'evict').
        result: Outcome ('hit', 'miss', 'success', 'error').
    """
    CACHE_OPERATIONS.labels(
        operation=operation,
        result=result,
    ).inc()


def record_pii_detection(
    pii_type: str,
    count: int = 1,
) -> None:
    """Record detected PII occurrences.

    Args:
        pii_type: Type of PII detected (e.g. 'SSN', 'EMAIL', 'CREDIT_CARD', 'API_KEY').
        count: Number of detections of this type.
    """
    PII_DETECTIONS.labels(
        pii_type=pii_type,
    ).inc(count)


def record_request_duration(
    provider: str,
    model: str,
    duration: float,
) -> None:
    """Record total request duration in seconds.

    Args:
        provider: Identifier of the LLM provider.
        model: Model identifier.
        duration: Total latency in seconds.
    """
    REQUEST_DURATION.labels(
        provider=provider,
        model=model,
    ).observe(duration)


def inc_active_requests(provider: str) -> None:
    """Increment the active in-flight request gauge for a provider.

    Args:
        provider: Identifier of the LLM provider.
    """
    ACTIVE_REQUESTS.labels(provider=provider).inc()


def dec_active_requests(provider: str) -> None:
    """Decrement the active in-flight request gauge for a provider.

    Args:
        provider: Identifier of the LLM provider.
    """
    ACTIVE_REQUESTS.labels(provider=provider).dec()


def get_metrics_output() -> tuple[bytes, str]:
    """Generate the latest Prometheus exposition format metrics payload.

    Returns:
        tuple[bytes, str]: Raw metrics payload bytes and the HTTP content type header.
    """
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "REQUESTS_TOTAL",
    "TTFT_SECONDS",
    "TOKENS_TOTAL",
    "CIRCUIT_STATE",
    "CACHE_OPERATIONS",
    "PII_DETECTIONS",
    "ACTIVE_REQUESTS",
    "REQUEST_DURATION",
    "CONTENT_TYPE_LATEST",
    "PROMETHEUS_AVAILABLE",
    "record_request",
    "record_ttft",
    "record_tokens",
    "update_circuit_state",
    "record_cache_operation",
    "record_pii_detection",
    "record_request_duration",
    "inc_active_requests",
    "dec_active_requests",
    "get_metrics_output",
]
