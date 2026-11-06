"""Main FastAPI application factory and entrypoint for LLM Gateway & Evaluation Proxy.

Configures lifespan events, asynchronous dependencies (Redis, embeddings, semantic cache),
resilience mechanisms (circuit breakers, intent routing), security middleware,
Prometheus metrics, and unified exception handlers.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.api.v1 import chat_router, health_router
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AllProvidersFailedError,
    AuthenticationError,
    CircuitOpenError,
    GatewayError,
    RateLimitError,
)
from app.core.logging import configure_logging, get_logger
from app.middleware.auth import APIKeyAuthMiddleware
from app.services.cache.redis_vector import RedisVectorStore
from app.services.cache.semantic_cache import SemanticCache
from app.services.circuit_breaker.breaker import CircuitBreakerRegistry
from app.services.embeddings.factory import EmbeddingServiceFactory
from app.services.providers.factory import ProviderFactory
from app.services.router.intent_router import IntentRouter
from app.services.sanitizer.pii import PIISanitizer
from app.telemetry.tracing import instrument_fastapi, setup_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle for startup initialization and graceful shutdown.

    Startup:
    1. Configure structured JSON logging.
    2. Initialize OpenTelemetry distributed tracing.
    3. Connect to Redis vector database.
    4. Initialize embedding service.
    5. Initialize semantic cache and index schema.
    6. Instantiate LLM provider adapters.
    7. Initialize circuit breaker registry.
    8. Instantiate intent router and PII sanitizer.
    9. Attach all components to app.state.

    Shutdown:
    1. Close semantic cache and Redis vector store connections.
    2. Close all provider HTTP sessions.
    3. Complete graceful teardown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control yielded back to running application.
    """
    settings: Settings = get_settings()

    # 1. Structured Logging
    configure_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting LLM Evaluation Proxy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
    )

    # 2. Distributed Tracing
    try:
        setup_tracing(settings.APP_NAME)
        instrument_fastapi(app)
    except Exception as trace_err:
        logger.warning("Tracing setup warning", error=str(trace_err))

    # 3. Redis Vector Store
    vector_store = RedisVectorStore(redis_url=settings.REDIS_URL)
    try:
        await vector_store.connect()
        logger.info("Redis vector store connection established", url=settings.REDIS_URL)
    except Exception as redis_err:
        logger.warning(
            "Redis connection failed during startup (will retry on requests)",
            error=str(redis_err),
        )

    # 4. Embedding Service
    try:
        embedding_service = EmbeddingServiceFactory.create(settings=settings)
        logger.info(
            "Embedding service initialized",
            provider=embedding_service.provider_name,
            dimensions=embedding_service.dimensions,
        )
    except Exception as emb_err:
        logger.warning("Embedding service initialization warning", error=str(emb_err))
        # Fallback to factory with defaults
        embedding_service = EmbeddingServiceFactory.create(
            settings=settings,
            provider="local",
            model_name=settings.EMBEDDING_MODEL_NAME,
        )

    # 5. Semantic Cache
    semantic_cache = SemanticCache(
        vector_store=vector_store,
        embedding_service=embedding_service,
        settings=settings,
    )
    try:
        await semantic_cache.initialize()
        logger.info("Semantic cache index initialized successfully")
    except Exception as cache_err:
        logger.warning(
            "Semantic cache index initialization failed (will retry on query)",
            error=str(cache_err),
        )

    # 6. LLM Providers
    providers = ProviderFactory.create_all(settings)
    logger.info("LLM provider adapters loaded", providers=list(providers.keys()))

    # 7. Circuit Breakers
    circuit_breakers = CircuitBreakerRegistry(settings=settings)

    # 8. Intent Router & PII Sanitizer
    intent_router = IntentRouter()
    pii_sanitizer = PIISanitizer()

    # 9. Store in Application State
    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.embedding_service = embedding_service
    app.state.semantic_cache = semantic_cache
    app.state.providers = providers
    app.state.circuit_breakers = circuit_breakers
    app.state.router = intent_router
    app.state.sanitizer = pii_sanitizer

    logger.info("Startup complete: LLM Proxy Gateway is ready to serve requests")

    yield

    # Shutdown Phase
    logger.info("Initiating graceful shutdown of LLM Proxy Gateway...")

    # 1. Close Semantic Cache & Vector Store
    try:
        await semantic_cache.close()
        logger.info("Semantic cache closed")
    except Exception as close_err:
        logger.warning("Error closing semantic cache", error=str(close_err))

    try:
        await vector_store.close()
        logger.info("Redis vector store closed")
    except Exception as close_err:
        logger.warning("Error closing Redis vector store", error=str(close_err))

    # 2. Close All Provider Sessions
    for p_name, provider in providers.items():
        try:
            await provider.close()
            logger.debug("Closed provider connection pool", provider=p_name)
        except Exception as p_close_err:
            logger.warning(
                "Error closing provider adapter", provider=p_name, error=str(p_close_err)
            )

    logger.info("Shutdown complete: all connections cleanly closed")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional Settings instance. Defaults to get_settings() singleton.

    Returns:
        FastAPI: Fully configured FastAPI application.
    """
    app_settings = settings or get_settings()

    app = FastAPI(
        title=app_settings.APP_NAME,
        version=app_settings.APP_VERSION,
        description="High-Throughput LLM Gateway & Evaluation Proxy with Semantic Caching and Resilience",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. Security & Authentication Middleware
    app.add_middleware(
        APIKeyAuthMiddleware,
        settings=app_settings,
    )

    # 2. Cross-Origin Resource Sharing (CORS) Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Mount Prometheus Metrics Endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # 4. Include API Routers
    app.include_router(chat_router)
    app.include_router(health_router)

    # 5. Global Exception Handlers
    @app.exception_handler(AuthenticationError)
    async def authentication_exception_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        logger.warning("Authentication failed", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "AUTHENTICATION_FAILED",
                "detail": exc.message,
                "code": exc.error_code,
            },
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(_request: Request, exc: RateLimitError) -> JSONResponse:
        logger.warning(
            "Rate limit exceeded",
            provider=exc.provider_name,
            error=str(exc),
            retry_after=exc.retry_after,
        )
        headers: dict[str, str] = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(int(exc.retry_after))
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "RATE_LIMIT_EXCEEDED",
                "detail": exc.message,
                "code": exc.error_code,
            },
            headers=headers,
        )

    @app.exception_handler(CircuitOpenError)
    async def circuit_open_exception_handler(
        _request: Request, exc: CircuitOpenError
    ) -> JSONResponse:
        logger.warning(
            "Circuit breaker open",
            provider=exc.provider_name,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "CIRCUIT_BREAKER_OPEN",
                "detail": exc.message,
                "code": exc.error_code,
            },
        )

    @app.exception_handler(AllProvidersFailedError)
    async def all_providers_failed_exception_handler(
        _request: Request, exc: AllProvidersFailedError
    ) -> JSONResponse:
        logger.error("All upstream providers failed", error=str(exc), details=exc.details)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ALL_PROVIDERS_FAILED",
                "detail": exc.message,
                "code": exc.error_code,
            },
        )

    @app.exception_handler(GatewayError)
    async def gateway_exception_handler(_request: Request, exc: GatewayError) -> JSONResponse:
        logger.error("Gateway error occurred", error=str(exc), error_code=exc.error_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "detail": exc.message,
                "code": exc.error_code,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled internal server error", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "detail": "An unexpected internal server error occurred.",
                "code": "INTERNAL_ERROR",
            },
        )

    return app


# Default singleton application instance
app = create_app()

__all__ = ["app", "create_app", "lifespan"]
