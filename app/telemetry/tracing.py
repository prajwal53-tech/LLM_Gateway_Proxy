"""OpenTelemetry Distributed Tracing Setup.

Configures the OpenTelemetry TracerProvider, tracer instances, and FastAPI middleware
instrumentation for distributed tracing across LLM gateway operations.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Tracer

    OPENTELEMETRY_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENTELEMETRY_AVAILABLE = False
    trace = None  # type: ignore
    SERVICE_NAME = "service.name"  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore

    class _FallbackSpan:
        """Fallback span context manager when opentelemetry is not installed."""

        def __enter__(self) -> "_FallbackSpan":
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def set_attribute(self, _key: str, _value: Any) -> None:
            pass

        def set_status(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def record_exception(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    class _FallbackTracer:
        """Fallback tracer when opentelemetry is not installed."""

        def __init__(self, name: str = "llm-eval-proxy") -> None:
            self.name = name

        @contextmanager
        def start_as_current_span(
            self,
            _name: str,
            *_args: Any,
            **_kwargs: Any,
        ) -> Generator[_FallbackSpan, None, None]:
            span = _FallbackSpan()
            yield span

        def start_span(
            self,
            _name: str,
            *_args: Any,
            **_kwargs: Any,
        ) -> _FallbackSpan:
            return _FallbackSpan()

    Tracer = _FallbackTracer  # type: ignore

from app.core.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger(__name__)

_TRACER_INITIALIZED: bool = False
_FALLBACK_TRACER_INSTANCE: Any = None


def setup_tracing(service_name: str = "llm-eval-proxy") -> Tracer:
    """Initialize OpenTelemetry tracer provider with service resource metadata.

    Creates and configures a TracerProvider with service attributes, registers
    it as the global OpenTelemetry tracer provider, and returns a tracer for the service.

    Args:
        service_name: Name of the application service (defaults to "llm-eval-proxy").

    Returns:
        Tracer: Configured OpenTelemetry Tracer instance.
    """
    global _TRACER_INITIALIZED, _FALLBACK_TRACER_INSTANCE

    if not OPENTELEMETRY_AVAILABLE:
        logger.info(
            "OpenTelemetry not installed; using fallback no-op tracer", service_name=service_name
        )
        _FALLBACK_TRACER_INSTANCE = _FallbackTracer(service_name)
        _TRACER_INITIALIZED = True
        return _FALLBACK_TRACER_INSTANCE

    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    _TRACER_INITIALIZED = True

    logger.info("OpenTelemetry tracing initialized", service_name=service_name)
    return trace.get_tracer(service_name)


def get_tracer(name: str | None = None) -> Tracer:
    """Retrieve an OpenTelemetry Tracer instance.

    If tracing has not been explicitly configured via setup_tracing(),
    this returns the current globally registered or default tracer.

    Args:
        name: Name of the tracer or instrumented module (defaults to "llm-eval-proxy").

    Returns:
        Tracer: Active OpenTelemetry Tracer instance.
    """
    tracer_name = name or "llm-eval-proxy"
    if not OPENTELEMETRY_AVAILABLE:
        return _FALLBACK_TRACER_INSTANCE or _FallbackTracer(tracer_name)
    return trace.get_tracer(tracer_name)


def instrument_fastapi(app: "FastAPI") -> None:
    """Instrument a FastAPI application instance with OpenTelemetry middleware.

    Automatically extracts incoming trace context and records server spans
    for all HTTP routes.

    Args:
        app: The FastAPI application instance to instrument.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="health,ready,metrics,docs,redoc,openapi.json",
        )
        logger.info("FastAPI OpenTelemetry instrumentation attached successfully")
    except Exception as exc:
        logger.warning(
            "OpenTelemetry FastAPI instrumentation not attached",
            detail=str(exc),
        )


__all__ = [
    "setup_tracing",
    "get_tracer",
    "instrument_fastapi",
    "OPENTELEMETRY_AVAILABLE",
]
