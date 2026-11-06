"""Unit tests for telemetry metrics and tracing modules."""

from fastapi import FastAPI

from app.telemetry.metrics import (
    dec_active_requests,
    get_metrics_output,
    inc_active_requests,
    record_cache_operation,
    record_pii_detection,
    record_request,
    record_request_duration,
    record_tokens,
    record_ttft,
    update_circuit_state,
)
from app.telemetry.tracing import (
    get_tracer,
    instrument_fastapi,
    setup_tracing,
)


def test_metrics_recording():
    """Test recording various Prometheus metric events."""
    # Test request counter
    record_request(
        provider="mock",
        model="gpt-4o",
        status="200",
        cache_hit=False,
        client_id="test_client",
    )

    # Test TTFT histogram
    record_ttft(provider="mock", model="gpt-4o", duration=0.125)

    # Test token counts
    record_tokens(
        provider="mock",
        model="gpt-4o",
        prompt_tokens=50,
        completion_tokens=25,
    )

    # Test circuit state gauge
    update_circuit_state(provider="mock", state_value=1)

    # Test cache operation
    record_cache_operation(operation="lookup", result="hit")

    # Test PII detection
    record_pii_detection(pii_type="SSN", count=2)

    # Test request duration
    record_request_duration(provider="mock", model="gpt-4o", duration=0.45)

    # Test active requests
    inc_active_requests(provider="mock")
    dec_active_requests(provider="mock")

    # Verify metrics output generation
    payload, content_type = get_metrics_output()
    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert "text/plain" in content_type


def test_tracing_setup():
    """Test OpenTelemetry tracing initialization and retrieval."""
    tracer = setup_tracing(service_name="test-llm-gateway")
    assert tracer is not None

    same_or_new_tracer = get_tracer("test_module")
    assert same_or_new_tracer is not None

    with tracer.start_as_current_span("test_span") as span:
        assert span is not None


def test_instrument_fastapi():
    """Test FastAPI instrumentation helper attaches without error."""
    app = FastAPI()
    instrument_fastapi(app)
