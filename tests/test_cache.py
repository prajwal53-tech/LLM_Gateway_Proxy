"""Unit tests for RedisVectorStore and SemanticCache."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.exceptions import ResponseError

from app.core.config import Settings
from app.core.exceptions import CacheError
from app.services.cache.redis_vector import RedisVectorStore
from app.services.cache.semantic_cache import SemanticCache
from app.services.embeddings.base import BaseEmbeddingService


class MockEmbeddingService(BaseEmbeddingService):
    """Test mock for BaseEmbeddingService."""

    @property
    def dimensions(self) -> int:
        return 384

    @property
    def provider_name(self) -> str:
        return "mock"

    async def embed(self, text: str) -> list[float]:
        del text
        return [0.1] * 384

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


class TestRedisVectorStore:
    """Test cases for RedisVectorStore."""

    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)

        store = RedisVectorStore(redis_client=mock_redis)
        await store.connect()
        mock_redis.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_index_success(self) -> None:
        mock_redis = MagicMock()
        mock_ft = MagicMock()
        mock_ft.create_index = AsyncMock()
        mock_redis.ft.return_value = mock_ft

        store = RedisVectorStore(redis_client=mock_redis)
        await store.initialize_index(dimensions=384)
        mock_ft.create_index.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_index_already_exists(self) -> None:
        mock_redis = MagicMock()
        mock_ft = MagicMock()
        mock_ft.create_index = AsyncMock(side_effect=ResponseError("Index already exists"))
        mock_redis.ft.return_value = mock_ft

        store = RedisVectorStore(redis_client=mock_redis)
        # Should not raise exception
        await store.initialize_index(dimensions=384)

    @pytest.mark.asyncio
    async def test_knn_search(self) -> None:
        mock_redis = MagicMock()
        mock_ft = MagicMock()

        mock_doc = MagicMock()
        mock_doc.id = "cache:abc"
        mock_doc.prompt = "What is Redis?"
        mock_doc.response = "Redis is an in-memory database."
        mock_doc.vector_distance = "0.04"

        mock_search_res = MagicMock(docs=[mock_doc])
        mock_ft.search = AsyncMock(return_value=mock_search_res)
        mock_redis.ft.return_value = mock_ft

        store = RedisVectorStore(redis_client=mock_redis)
        results = await store.knn_search(query_vector=[0.1] * 384, k=1)

        assert len(results) == 1
        assert results[0]["id"] == "cache:abc"
        assert results[0]["prompt"] == "What is Redis?"
        assert results[0]["response"] == "Redis is an in-memory database."
        assert results[0]["distance"] == 0.04

    @pytest.mark.asyncio
    async def test_store_and_expire(self) -> None:
        mock_redis = MagicMock()
        mock_json = MagicMock()
        mock_json.set = AsyncMock()
        mock_redis.json.return_value = mock_json
        mock_redis.expire = AsyncMock()

        store = RedisVectorStore(redis_client=mock_redis)
        await store.store(
            key="cache:123",
            prompt="Hello",
            response="World",
            embedding=[0.1] * 384,
            ttl=3600,
        )

        mock_json.set.assert_awaited_once()
        mock_redis.expire.assert_awaited_once_with("cache:123", 3600)


class TestSemanticCache:
    """Test cases for SemanticCache."""

    @pytest.fixture
    def mock_embedding_service(self) -> BaseEmbeddingService:
        return MockEmbeddingService()

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            CACHE_SIMILARITY_THRESHOLD=0.08,
            CACHE_TTL_SECONDS=86400,
        )

    @pytest.mark.asyncio
    async def test_initialize(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        mock_store.initialize_index = AsyncMock()

        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        await cache.initialize()
        mock_store.initialize_index.assert_awaited_once_with(dimensions=384)

    @pytest.mark.asyncio
    async def test_lookup_hit(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        mock_store.knn_search = AsyncMock(
            return_value=[
                {
                    "id": "cache:1",
                    "prompt": "test",
                    "response": "Cached answer",
                    "distance": 0.05,
                }
            ]
        )

        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        result = await cache.lookup("test")
        assert result == "Cached answer"

    @pytest.mark.asyncio
    async def test_lookup_miss_distance_exceeded(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        mock_store.knn_search = AsyncMock(
            return_value=[
                {
                    "id": "cache:1",
                    "prompt": "test",
                    "response": "Far away answer",
                    "distance": 0.15,
                }
            ]
        )

        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        result = await cache.lookup("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_empty_prompt(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        result = await cache.lookup("")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_success(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        mock_store.store = AsyncMock()

        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        await cache.store(prompt="hello", response="world")
        mock_store.store.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_failure_raises_cache_error(
        self, mock_embedding_service: BaseEmbeddingService, settings: Settings
    ) -> None:
        mock_store = MagicMock(spec=RedisVectorStore)
        mock_store.store = AsyncMock(side_effect=Exception("Redis failure"))

        cache = SemanticCache(
            vector_store=mock_store,
            embedding_service=mock_embedding_service,
            settings=settings,
        )
        with pytest.raises(CacheError):
            await cache.store(prompt="hello", response="world")
