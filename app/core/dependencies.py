"""FastAPI dependency injection providers.

Supplies application settings, Redis client connections, embedding service instances,
semantic cache, and PII sanitizers to route handlers with proper lifecycle management.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from fastapi import Depends

from app.core.config import Settings, get_settings

if TYPE_CHECKING:
    from app.services.cache.semantic_cache import SemanticCache
    from app.services.embeddings.base import BaseEmbeddingService
    from app.services.sanitizer.pii_sanitizer import PIISanitizer


async def get_redis_client(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[aioredis.Redis, None]:
    """Provide an asynchronous Redis client connection with automatic cleanup.

    Args:
        settings: Injected application settings containing REDIS_URL.

    Yields:
        aioredis.Redis: An active Redis client session.
    """
    client: aioredis.Redis = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=False,
    )
    try:
        yield client
    finally:
        await client.aclose()


@lru_cache
def _get_cached_embedding_service(
    provider: str,
    model_name: str,
) -> "BaseEmbeddingService":
    """Internal factory cache for embedding service instances.

    Args:
        provider: Provider identifier ('local', 'openai', etc.).
        model_name: Name of the embedding model.

    Returns:
        BaseEmbeddingService: Instantiated embedding service.
    """
    from app.services.embeddings.factory import EmbeddingServiceFactory

    return EmbeddingServiceFactory.create(
        provider=provider,
        model_name=model_name,
    )


def get_embedding_service(
    settings: Settings = Depends(get_settings),
) -> "BaseEmbeddingService":
    """FastAPI dependency providing the configured embedding service.

    Args:
        settings: Injected application settings.

    Returns:
        BaseEmbeddingService: Configured embedding provider instance.
    """
    return _get_cached_embedding_service(
        provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL_NAME,
    )


async def get_semantic_cache(
    settings: Settings = Depends(get_settings),
    redis_client: aioredis.Redis = Depends(get_redis_client),
    embedding_service: "BaseEmbeddingService" = Depends(get_embedding_service),
) -> "SemanticCache":
    """FastAPI dependency providing initialized SemanticCache service.

    Args:
        settings: Injected application settings.
        redis_client: Injected Redis client connection.
        embedding_service: Injected embedding service.

    Returns:
        SemanticCache: Semantic cache instance configured for Redis Vector Search.
    """
    from app.services.cache.semantic_cache import SemanticCache

    return SemanticCache(
        redis_client=redis_client,
        embedding_service=embedding_service,
        ttl_seconds=settings.CACHE_TTL_SECONDS,
        similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
    )


@lru_cache
def get_pii_sanitizer() -> "PIISanitizer":
    """FastAPI dependency providing singleton PIISanitizer instance.

    Returns:
        PIISanitizer: Regex-based PII detection and redaction instance.
    """
    from app.services.sanitizer.pii_sanitizer import PIISanitizer

    return PIISanitizer()


__all__ = [
    "get_settings",
    "get_redis_client",
    "get_embedding_service",
    "get_semantic_cache",
    "get_pii_sanitizer",
]
