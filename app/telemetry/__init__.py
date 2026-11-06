"""Telemetry package providing Prometheus metrics and OpenTelemetry tracing."""

from app.telemetry.metrics import (
    ACTIVE_REQUESTS,
    CACHE_OPERATIONS,
    CIRCUIT_STATE,
    CONTENT_TYPE_LATEST,
    PII_DETECTIONS,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    TTFT_SECONDS,
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

__all__ = [
    # Metrics
    "REQUESTS_TOTAL",
    "TTFT_SECONDS",
    "TOKENS_TOTAL",
    "CIRCUIT_STATE",
    "CACHE_OPERATIONS",
    "PII_DETECTIONS",
    "ACTIVE_REQUESTS",
    "REQUEST_DURATION",
    "CONTENT_TYPE_LATEST",
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
    # Tracing
    "setup_tracing",
    "get_tracer",
    "instrument_fastapi",
]
