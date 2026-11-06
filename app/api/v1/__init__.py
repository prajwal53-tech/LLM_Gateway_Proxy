"""V1 API Router Package.

Exports chat completions and health check routers.
"""

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router

__all__ = ["chat_router", "health_router"]
