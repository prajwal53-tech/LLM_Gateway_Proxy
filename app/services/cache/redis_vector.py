"""Redis Vector Store implementation using RediSearch HNSW and JSON documents.

Provides index creation, vector similarity KNN search with DIALECT 2,
and document storage with configurable TTL.
"""

import asyncio
import json
import struct
from typing import Any

import redis.asyncio as aioredis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.core.exceptions import CacheError
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisVectorStore:
    """Redis Stack vector store managing HNSW indexes and KNN vector search on JSON."""

    INDEX_NAME: str = "idx:llm_cache"
    PREFIX: str = "cache:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_client: aioredis.Redis | None = None,
        index_name: str = "idx:llm_cache",
        prefix: str = "cache:",
    ) -> None:
        """Initialize RedisVectorStore client and configuration.

        Args:
            redis_url: Redis Stack connection string URL.
            redis_client: Optional pre-existing aioredis.Redis instance.
            index_name: RediSearch index identifier.
            prefix: Key prefix for cached JSON documents.
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.prefix = prefix

        if redis_client is not None:
            self._redis = redis_client
            self._owns_client = False
        else:
            self._redis = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
            self._owns_client = True

    @property
    def redis(self) -> aioredis.Redis:
        """Active Redis client instance."""
        return self._redis

    async def connect(self) -> None:
        """Establish connection and ping Redis.

        Raises:
            CacheError: If ping fails or connection cannot be established.
        """
        try:
            await self._redis.ping()
            logger.info("redis_connected", redis_url=self.redis_url)
        except Exception as e:
            logger.error("redis_connection_failed", redis_url=self.redis_url, error=str(e))
            raise CacheError(f"Failed to connect to Redis at {self.redis_url}: {e}") from e

    async def initialize_index(self, dimensions: int) -> None:
        """Create RediSearch HNSW vector index on JSON documents if not exists.

        Configures fields:
        - prompt: TextField for prompt query text
        - response: TextField for LLM output text
        - embedding: VectorField (HNSW, FLOAT32, COSINE distance, dynamic DIM)

        Args:
            dimensions: Dimensionality of vector embeddings.

        Raises:
            CacheError: If index creation fails.
        """
        try:
            schema = (
                TextField("$.prompt", as_name="prompt"),
                TextField("$.response", as_name="response"),
                VectorField(
                    "$.embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dimensions,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            )
            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.JSON,
            )
            await self._redis.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )
            logger.info(
                "redis_vector_index_created",
                index_name=self.index_name,
                dimensions=dimensions,
                prefix=self.prefix,
            )
        except ResponseError as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "index already exists" in error_msg:
                logger.info(
                    "redis_vector_index_already_exists",
                    index_name=self.index_name,
                )
            else:
                logger.error(
                    "redis_index_creation_failed",
                    index_name=self.index_name,
                    error=str(e),
                )
                raise CacheError(
                    f"Failed to create Redis vector index '{self.index_name}': {e}"
                ) from e
        except Exception as e:
            logger.error(
                "redis_index_creation_unexpected_error",
                index_name=self.index_name,
                error=str(e),
            )
            raise CacheError(f"Unexpected error creating Redis vector index: {e}") from e

    async def knn_search(
        self,
        query_vector: list[float],
        k: int = 1,
    ) -> list[dict[str, Any]]:
        """Perform KNN vector search against RediSearch index using DIALECT 2.

        Args:
            query_vector: Query embedding float vector.
            k: Number of nearest neighbors to retrieve.

        Returns:
            list[dict[str, Any]]: List of match results with keys:
                'id', 'prompt', 'response', 'distance'.

        Raises:
            CacheError: If vector search execution fails.
        """
        if not query_vector:
            return []

        try:
            vector_bytes = struct.pack(f"{len(query_vector)}f", *query_vector)
            query_str = f"*=>[KNN {k} @embedding $vector AS vector_distance]"

            query = (
                Query(query_str)
                .sort_by("vector_distance")
                .return_fields("prompt", "response", "vector_distance")
                .paging(0, k)
                .dialect(2)
            )
            params = {"vector": vector_bytes}

            search_result = await self._redis.ft(self.index_name).search(
                query=query,
                query_params=params,
            )

            matches: list[dict[str, Any]] = []
            for doc in getattr(search_result, "docs", []):
                dist_raw = getattr(doc, "vector_distance", 0.0)
                try:
                    distance = float(dist_raw)
                except (ValueError, TypeError):
                    distance = 0.0

                prompt_raw = getattr(doc, "prompt", "")
                response_raw = getattr(doc, "response", "")

                # Decode bytes if needed
                prompt_val = (
                    prompt_raw.decode("utf-8") if isinstance(prompt_raw, bytes) else str(prompt_raw)
                )
                response_val = (
                    response_raw.decode("utf-8")
                    if isinstance(response_raw, bytes)
                    else str(response_raw)
                )
                doc_id = doc.id.decode("utf-8") if isinstance(doc.id, bytes) else str(doc.id)

                matches.append(
                    {
                        "id": doc_id,
                        "prompt": prompt_val,
                        "response": response_val,
                        "distance": distance,
                    }
                )
            return matches
        except Exception as e:
            logger.error(
                "redis_knn_search_failed",
                index_name=self.index_name,
                error=str(e),
            )
            raise CacheError(f"Redis KNN vector search failed: {e}") from e

    async def store(
        self,
        key: str,
        prompt: str,
        response: str,
        embedding: list[float],
        ttl: int,
    ) -> None:
        """Store prompt, completion response, and embedding vector as a JSON document.

        Args:
            key: Redis document key (e.g. 'cache:<sha256>').
            prompt: Original user prompt string.
            response: LLM completion response string.
            embedding: Normalized embedding vector.
            ttl: Entry time-to-live in seconds (0 disables expiration).

        Raises:
            CacheError: If document persistence fails.
        """
        doc_key = key if key.startswith(self.prefix) else f"{self.prefix}{key}"
        doc_data = {
            "prompt": prompt,
            "response": response,
            "embedding": embedding,
        }

        try:
            if hasattr(self._redis, "json"):
                await self._redis.json().set(doc_key, "$", doc_data)
            else:
                await self._redis.execute_command(
                    "JSON.SET",
                    doc_key,
                    "$",
                    json.dumps(doc_data),
                )

            if ttl > 0:
                await self._redis.expire(doc_key, ttl)

            logger.debug("redis_vector_doc_stored", key=doc_key, ttl=ttl)
        except Exception as e:
            logger.error("redis_store_failed", key=doc_key, error=str(e))
            raise CacheError(f"Failed to store vector document at key '{doc_key}': {e}") from e

    async def close(self) -> None:
        """Close Redis connection if owned by this instance."""
        if not self._owns_client:
            return

        try:
            if hasattr(self._redis, "aclose"):
                await self._redis.aclose()
            elif hasattr(self._redis, "close"):
                res = self._redis.close()
                if asyncio.iscoroutine(res):
                    await res
            logger.info("redis_connection_closed")
        except Exception as e:
            logger.warning("redis_close_error", error=str(e))
