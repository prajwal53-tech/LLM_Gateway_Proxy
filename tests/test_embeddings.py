"""Unit tests for embedding services and factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import CacheError, ProviderAPIError
from app.services.embeddings.factory import EmbeddingServiceFactory
from app.services.embeddings.gemini_embed import GeminiEmbeddingService
from app.services.embeddings.local import LocalEmbeddingService, _EmbeddingLRUCache
from app.services.embeddings.openai_embed import OpenAIEmbeddingService


class TestEmbeddingLRUCache:
    """Test cases for internal LRU embedding cache."""

    def test_cache_get_set_and_eviction(self) -> None:
        cache = _EmbeddingLRUCache(maxsize=2)
        assert len(cache) == 0

        cache.set("k1", [0.1, 0.2])
        cache.set("k2", [0.3, 0.4])
        assert len(cache) == 2
        # Touch k1 to make k2 the least recently used
        assert cache.get("k1") == [0.1, 0.2]

        # Insert k3, k2 should be evicted
        cache.set("k3", [0.5, 0.6])
        assert len(cache) == 2
        assert cache.get("k1") == [0.1, 0.2]
        assert cache.get("k3") == [0.5, 0.6]
        assert cache.get("k2") is None

        cache.clear()
        assert len(cache) == 0
        assert cache.get("k1") is None


class TestLocalEmbeddingService:
    """Test cases for LocalEmbeddingService."""

    @pytest.mark.asyncio
    async def test_embed_single_with_mock_model(self) -> None:
        service = LocalEmbeddingService(model_name="all-MiniLM-L6-v2", cache_size=100)
        assert service.dimensions == 384
        assert service.provider_name == "local"

        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 384
        service._model = mock_model

        result1 = await service.embed("hello world")
        assert len(result1) == 384
        assert mock_model.encode.call_count == 1

        # Second call should hit the LRU cache
        result2 = await service.embed("hello world")
        assert result2 == result1
        assert mock_model.encode.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch_with_caching(self) -> None:
        service = LocalEmbeddingService(model_name="all-MiniLM-L6-v2", cache_size=100)
        mock_model = MagicMock()

        def mock_encode(data, **_kwargs):
            if isinstance(data, list):
                return [[0.5] * 384 for _ in data]
            return [0.5] * 384

        mock_model.encode.side_effect = mock_encode
        service._model = mock_model

        # Pre-cache one item
        await service.embed("item1")
        assert mock_model.encode.call_count == 1

        # Batch with pre-cached and new item
        results = await service.embed_batch(["item1", "item2"])
        assert len(results) == 2
        assert len(results[0]) == 384
        assert len(results[1]) == 384
        # encode called once for single and once for uncached batch ("item2")
        assert mock_model.encode.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self) -> None:
        service = LocalEmbeddingService()
        results = await service.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_model_load_failure(self) -> None:
        service = LocalEmbeddingService()
        with patch.object(service, "_get_model", side_effect=Exception("Model load error")):
            with pytest.raises(CacheError) as exc_info:
                await service.embed("test")
            assert "Local embedding failed" in str(exc_info.value)


class TestOpenAIEmbeddingService:
    """Test cases for OpenAIEmbeddingService."""

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        mock_client = MagicMock()
        mock_create = AsyncMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.01] * 1536
        mock_response = MagicMock(data=[mock_item])
        mock_create.return_value = mock_response
        mock_client.embeddings.create = mock_create

        service = OpenAIEmbeddingService(
            api_key="sk-test",
            model_name="text-embedding-3-small",
            dimensions=1536,
            client=mock_client,
        )
        assert service.dimensions == 1536
        assert service.provider_name == "openai"

        result = await service.embed("test prompt")
        assert len(result) == 1536
        assert result == [0.01] * 1536
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_embed_batch_sorted(self) -> None:
        mock_client = MagicMock()
        mock_create = AsyncMock()

        item0 = MagicMock(index=0, embedding=[0.1] * 1536)
        item1 = MagicMock(index=1, embedding=[0.2] * 1536)
        # Return out of order
        mock_response = MagicMock(data=[item1, item0])
        mock_create.return_value = mock_response
        mock_client.embeddings.create = mock_create

        service = OpenAIEmbeddingService(client=mock_client)
        results = await service.embed_batch(["text1", "text2"])
        assert len(results) == 2
        assert results[0] == [0.1] * 1536
        assert results[1] == [0.2] * 1536

    @pytest.mark.asyncio
    async def test_embed_error_handling(self) -> None:
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        service = OpenAIEmbeddingService(client=mock_client)
        with pytest.raises(ProviderAPIError) as exc_info:
            await service.embed("fail text")
        assert "OpenAI embedding generation failed" in str(exc_info.value)


class TestGeminiEmbeddingService:
    """Test cases for GeminiEmbeddingService."""

    @pytest.mark.asyncio
    async def test_embed_single_async_client(self) -> None:
        mock_client = MagicMock()
        mock_embed_fn = AsyncMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.02] * 768
        mock_response = MagicMock(embedding=mock_emb, embeddings=[mock_emb])
        mock_embed_fn.return_value = mock_response

        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = mock_embed_fn

        service = GeminiEmbeddingService(
            api_key="gemini-key",
            model_name="text-embedding-004",
            dimensions=768,
            client=mock_client,
        )
        assert service.dimensions == 768
        assert service.provider_name == "gemini"

        result = await service.embed("gemini text")
        assert len(result) == 768
        assert result == [0.02] * 768

    @pytest.mark.asyncio
    async def test_embed_batch_async_client(self) -> None:
        mock_client = MagicMock()
        mock_embed_fn = AsyncMock()
        mock_emb1 = MagicMock(values=[0.1] * 768)
        mock_emb2 = MagicMock(values=[0.2] * 768)
        mock_response = MagicMock(embeddings=[mock_emb1, mock_emb2])
        mock_embed_fn.return_value = mock_response

        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.embed_content = mock_embed_fn

        service = GeminiEmbeddingService(client=mock_client)
        results = await service.embed_batch(["text 1", "text 2"])
        assert len(results) == 2
        assert results[0] == [0.1] * 768
        assert results[1] == [0.2] * 768


class TestEmbeddingServiceFactory:
    """Test cases for EmbeddingServiceFactory."""

    def setup_method(self) -> None:
        EmbeddingServiceFactory.reset()

    def test_create_local_singleton(self) -> None:
        settings = Settings(EMBEDDING_PROVIDER="local", EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2")
        service1 = EmbeddingServiceFactory.create(settings=settings)
        service2 = EmbeddingServiceFactory.create(settings=settings)

        assert isinstance(service1, LocalEmbeddingService)
        assert service1 is service2
        assert service1.dimensions == 384

    def test_create_openai(self) -> None:
        settings = Settings(
            EMBEDDING_PROVIDER="openai",
            EMBEDDING_MODEL_NAME="text-embedding-3-small",
            OPENAI_API_KEY="sk-test",
        )
        service = EmbeddingServiceFactory.create(settings=settings)
        assert isinstance(service, OpenAIEmbeddingService)
        assert service.dimensions == 1536

    def test_create_gemini(self) -> None:
        settings = Settings(
            EMBEDDING_PROVIDER="gemini",
            EMBEDDING_MODEL_NAME="text-embedding-004",
            GEMINI_API_KEY="test-gemini",
        )
        service = EmbeddingServiceFactory.create(settings=settings)
        assert isinstance(service, GeminiEmbeddingService)
        assert service.dimensions == 768

    def test_unsupported_provider_raises(self) -> None:
        settings = Settings(EMBEDDING_PROVIDER="invalid_provider")
        with pytest.raises(ValueError) as exc_info:
            EmbeddingServiceFactory.create(settings=settings)
        assert "Unsupported embedding provider" in str(exc_info.value)
