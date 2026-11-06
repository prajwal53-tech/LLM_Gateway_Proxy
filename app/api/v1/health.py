"""Health check and readiness probe endpoints.

Provides service health status, Redis connectivity verification,
and embedding model loading verification for Kubernetes and monitoring systems.
"""

from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.api.v1.schemas import HealthResponse
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


async def _check_redis(request: Request) -> str:
    """Check Redis connectivity from application state.

    Args:
        request: Incoming FastAPI request.

    Returns:
        str: "connected" if Redis is reachable, "disconnected" otherwise.
    """
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is not None:
        try:
            if hasattr(vector_store, "redis") and vector_store.redis is not None:
                await vector_store.redis.ping()
                return "connected"
            if hasattr(vector_store, "connect"):
                await vector_store.connect()
                return "connected"
        except Exception as exc:
            logger.warning("Redis health check ping failed", error=str(exc))
            return "disconnected"
    return "disconnected"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Check overall proxy service health status and Redis connectivity.",
)
@router.get(
    "/v1/health",
    response_model=HealthResponse,
    include_in_schema=False,
)
async def health_check(request: Request) -> HealthResponse:
    """Perform health verification across critical dependencies.

    Args:
        request: Incoming FastAPI request.

    Returns:
        HealthResponse: Health status, Redis state, and current service version.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    redis_status = await _check_redis(request)
    service_status = "healthy" if redis_status == "connected" else "degraded"

    return HealthResponse(
        status=service_status,
        redis=redis_status,
        version=settings.APP_VERSION,
    )


@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Kubernetes readiness probe verifying Redis connectivity and embedding model availability.",
    response_model=None,
)
@router.get(
    "/v1/ready",
    include_in_schema=False,
    response_model=None,
)
async def readiness_check(request: Request) -> JSONResponse | dict[str, Any]:
    """Verify that all core components are initialized and ready to accept traffic.

    Args:
        request: Incoming FastAPI request.

    Returns:
        JSONResponse | dict: Readiness payload indicating component statuses.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    redis_status = await _check_redis(request)

    embedding_service = getattr(request.app.state, "embedding_service", None)
    embedding_loaded = embedding_service is not None

    is_ready = (redis_status == "connected") and embedding_loaded

    payload: dict[str, Any] = {
        "status": "ready" if is_ready else "not_ready",
        "redis": redis_status,
        "embedding_service": "loaded" if embedding_loaded else "not_loaded",
        "version": settings.APP_VERSION,
    }

    if not is_ready:
        logger.warning(
            "Readiness probe failed",
            redis=redis_status,
            embedding_service="loaded" if embedding_loaded else "not_loaded",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload,
        )

    return payload


__all__ = ["router", "health_check", "readiness_check"]
