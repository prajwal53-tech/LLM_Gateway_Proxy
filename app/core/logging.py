"""Structured logging configuration using structlog.

Provides JSON-formatted structured logging with context propagation, timestamps,
log levels, and standard library logging integration.
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application using structlog.

    Sets up processors for ISO timestamps, log levels, exception tracebacks,
    context variable merging, and JSON rendering. Also configures standard
    library logging handlers to route through structlog.

    Args:
        log_level: The root logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Processors shared between structlog and standard library logging
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Silence noisy third-party loggers if needed
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        log = logging.getLogger(uvicorn_logger)
        log.handlers.clear()
        log.addHandler(handler)
        log.propagate = False


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retrieve a structured logger instance with an optional logger name.

    Args:
        name: Name of the logger, typically __name__.

    Returns:
        structlog.stdlib.BoundLogger: Configured bound logger instance.
    """
    return structlog.get_logger(name or "app")
