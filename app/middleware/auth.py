"""API Key Authentication Middleware.

Enforces API key validation for incoming HTTP requests against configured gateway
keys. Attaches tenant/client identity to request state on success and bypasses
exempt health and metrics endpoints.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Endpoints exempt from API key authentication
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for authenticating requests via the X-API-Key header.

    Attributes:
        settings: Application settings containing auth toggles and API key mappings.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the APIKeyAuthMiddleware.

        Args:
            app: The ASGI application instance.
            settings: Optional application settings. Defaults to get_settings() singleton.
        """
        super().__init__(app)
        self.settings = settings or get_settings()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process incoming request, authenticate via API key, and delegate to next handler.

        Args:
            request: The incoming Starlette/FastAPI HTTP request.
            call_next: The next middleware or route handler endpoint.

        Returns:
            Response: HTTP response from the downstream handler or a 401 Unauthorized JSONResponse.
        """
        path = request.url.path.rstrip("/")
        normalized_path = path if path else "/"

        # 1. Skip auth for health, readiness, metrics, and documentation endpoints
        if (
            request.url.path in EXEMPT_PATHS
            or normalized_path in EXEMPT_PATHS
            or any(request.url.path.startswith(f"{exempt}/") for exempt in EXEMPT_PATHS)
        ):
            return await call_next(request)

        # 2. Bypass authentication if disabled globally in configuration
        if not self.settings.AUTH_ENABLED:
            request.state.client_id = "dev_bypass"
            return await call_next(request)

        # 3. Extract X-API-Key header (case-insensitive lookup via Starlette Headers)
        api_key = request.headers.get("X-API-Key")

        # 4. Validate API key against configured gateway keys
        if not api_key or api_key not in self.settings.GATEWAY_API_KEYS:
            logger.warning(
                "Authentication failed: invalid or missing API key",
                path=request.url.path,
                method=request.method,
                has_key=bool(api_key),
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Invalid or missing API key",
                },
            )

        # 5. Attach resolved client/tenant ID to request state
        client_id = self.settings.GATEWAY_API_KEYS[api_key]
        request.state.client_id = client_id

        logger.info(
            "Authentication successful",
            client_id=client_id,
            path=request.url.path,
            method=request.method,
        )

        # 6. Proceed to downstream handler
        response: Response = await call_next(request)
        return response


__all__ = ["APIKeyAuthMiddleware", "EXEMPT_PATHS"]
