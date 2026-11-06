"""Middleware package for the LLM evaluation gateway."""

from app.middleware.auth import APIKeyAuthMiddleware

__all__ = ["APIKeyAuthMiddleware"]
