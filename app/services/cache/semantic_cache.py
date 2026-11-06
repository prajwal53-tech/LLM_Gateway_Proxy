"""Semantic caching service utilizing Redis vector similarity search.

Provides prompt embedding, vector similarity lookup with threshold filtering,
and document storage with TTL expiration.
"""

import hashlib
import time
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import CacheError
from app.core.logging import get_logger
from app.services.cache.redis_vector import RedisVectorStore
from app.services.embeddings.base import BaseEmbeddingService

logger = get_logger(__name__)


class SemanticCache:
    """High-throughput semantic cache backed by Redis vector similarity search."""

    def __init__(
        self,
        vector_store: RedisVectorStore | None = None,
        embedding_service: BaseEmbeddingService | None = None,
        settings: Settings | None = None,
        redis_client: Any | None = None,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        """Initialize SemanticCache with vector store and embedding service.

        Args:
            vector_store: Pre-configured RedisVectorStore instance.
            embedding_service: Embedding service for prompt vectorization.
            settings: Optional application settings.
            redis_client: Optional raw aioredis.Redis instance (used if vector_store is None).
            redis_url: Optional Redis URL string (used if vector_store and redis_client are None).
            ttl_seconds: Optional cache TTL override in seconds.
            similarity_threshold: Optional cosine distance threshold override.
        """
        app_settings = settings or get_settings()

        if vector_store is not None:
            self.vector_store = vector_store
        elif redis_client is not None:
            self.vector_store = RedisVectorStore(redis_client=redis_client)
        else:
            url = redis_url or app_settings.REDIS_URL
            self.vector_store = RedisVectorStore(redis_url=url)

        if embedding_service is not None:
            self.embedding_service = embedding_service
        else:
            from app.services.embeddings.factory import EmbeddingServiceFactory

            self.embedding_service = EmbeddingServiceFactory.create(settings=app_settings)

        self.ttl_seconds: int = (
            ttl_seconds if ttl_seconds is not None else app_settings.CACHE_TTL_SECONDS
        )
        self.similarity_threshold: float = (
            similarity_threshold
            if similarity_threshold is not None
            else app_settings.CACHE_SIMILARITY_THRESHOLD
        )

    async def initialize(self) -> None:
        """Initialize Redis vector index using the embedding service dimensions.

        Raises:
            CacheError: If vector store index initialization fails.
        """
        dimensions = self.embedding_service.dimensions
        logger.info(
            "initializing_semantic_cache_index",
            dimensions=dimensions,
            provider=self.embedding_service.provider_name,
        )
        await self.vector_store.initialize_index(dimensions=dimensions)

    async def lookup(self, prompt: str) -> str | None:
        """Look up a semantic cache entry for the given prompt.

        Embeds the input prompt, executes a KNN search with k=1 in Redis,
        and returns the cached response if the cosine distance is within
        the configured similarity threshold.

        Args:
            prompt: User input prompt string.

        Returns:
            str | None: Cached response text on hit, or None on miss / error.
        """
        if not prompt or not prompt.strip():
            return None

        start_time = time.perf_counter()
        clean_prompt = prompt.strip()

        try:
            query_vector = await self.embedding_service.embed(clean_prompt)
            results = await self.vector_store.knn_search(query_vector=query_vector, k=1)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            if results:
                best_match = results[0]
                distance = float(best_match.get("distance", 1.0))

                if distance <= self.similarity_threshold:
                    logger.info(
                        "semantic_cache_hit",
                        distance=round(distance, 5),
                        threshold=self.similarity_threshold,
                        duration_ms=round(duration_ms, 2),
                        cached_id=best_match.get("id"),
                    )
                    return str(best_match.get("response", ""))

                logger.info(
                    "semantic_cache_miss_distance_exceeded",
                    distance=round(distance, 5),
                    threshold=self.similarity_threshold,
                    duration_ms=round(duration_ms, 2),
                )
                return None

            logger.info(
                "semantic_cache_miss_no_results",
                duration_ms=round(duration_ms, 2),
            )
            return None
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "semantic_cache_lookup_error",
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            # Treat errors during lookup as cache miss to avoid blocking requests
            return None

    async def store(self, prompt: str, response: str) -> None:
        """Embed and persist a prompt-response pair into the semantic cache with TTL.

        Args:
            prompt: User input prompt string.
            response: LLM completion text to cache.

        Raises:
            CacheError: If embedding or Redis storage fails.
        """
        if not prompt or not response:
            return

        start_time = time.perf_counter()
        clean_prompt = prompt.strip()

        try:
            embedding = await self.embedding_service.embed(clean_prompt)
            prompt_hash = hashlib.sha256(clean_prompt.encode("utf-8")).hexdigest()
            doc_key = f"cache:{prompt_hash}"

            await self.vector_store.store(
                key=doc_key,
                prompt=clean_prompt,
                response=response,
                embedding=embedding,
                ttl=self.ttl_seconds,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "semantic_cache_stored",
                key=doc_key,
                ttl=self.ttl_seconds,
                duration_ms=round(duration_ms, 2),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "semantic_cache_store_failed",
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            if isinstance(e, CacheError):
                raise
            raise CacheError(f"Failed to store entry in semantic cache: {e}") from e

    async def close(self) -> None:
        """Close underlying vector store connection."""
        await self.vector_store.close()
