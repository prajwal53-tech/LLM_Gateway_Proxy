"""Local text embedding service utilizing SentenceTransformers.

Provides local high-throughput embeddings using the all-MiniLM-L6-v2 model (384 dims),
non-blocking asyncio thread execution, and in-memory LRU caching of embeddings.
"""

import asyncio
import hashlib
import threading
from collections import OrderedDict
from typing import Any

from app.core.exceptions import CacheError
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingService

logger = get_logger(__name__)


class _EmbeddingLRUCache:
    """Thread-safe in-memory LRU cache for prompt embeddings."""

    def __init__(self, maxsize: int = 10000) -> None:
        """Initialize LRU cache with maximum entry capacity.

        Args:
            maxsize: Maximum number of embeddings to retain.
        """
        self._maxsize = maxsize
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> list[float] | None:
        """Retrieve embedding from cache if present.

        Args:
            key: Hash string identifier for the text.

        Returns:
            list[float] | None: Cached embedding vector or None.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: list[float]) -> None:
        """Store embedding in cache, evicting oldest item if maxsize exceeded.

        Args:
            key: Hash string identifier for the text.
            value: Float vector embedding.
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                return
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return current number of items in cache."""
        with self._lock:
            return len(self._cache)


class LocalEmbeddingService(BaseEmbeddingService):
    """Local embedding service powered by SentenceTransformer all-MiniLM-L6-v2."""

    _dimensions: int = 384
    _provider_name: str = "local"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 10000,
    ) -> None:
        """Initialize LocalEmbeddingService with lazy model loading and LRU cache.

        Args:
            model_name: HuggingFace model name or path.
            cache_size: Maximum number of embeddings to keep in LRU cache.
        """
        self.model_name = model_name
        self._lru_cache = _EmbeddingLRUCache(maxsize=cache_size)
        self._model: Any = None
        self._load_lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        """Vector dimension for all-MiniLM-L6-v2 (384)."""
        return self._dimensions

    @property
    def provider_name(self) -> str:
        """Provider identifier string ('local')."""
        return self._provider_name

    def _get_model(self) -> Any:
        """Lazy-load the SentenceTransformer model in a thread-safe manner."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("loading_local_embedding_model", model=self.model_name)
                        self._model = SentenceTransformer(self.model_name)
                        logger.info("local_embedding_model_loaded", model=self.model_name)
                    except Exception as e:
                        logger.error(
                            "local_embedding_model_load_failed",
                            model=self.model_name,
                            error=str(e),
                        )
                        raise CacheError(
                            f"Failed to load local embedding model '{self.model_name}': {e}"
                        ) from e
        return self._model

    def _hash_text(self, text: str) -> str:
        """Generate SHA-256 hash key for text caching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encode_single_sync(self, text: str) -> list[float]:
        """Synchronously encode and normalize a single text string."""
        model = self._get_model()
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)

    def _encode_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronously encode and normalize a batch of text strings."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(emb) for emb in embeddings]

    async def embed(self, text: str) -> list[float]:
        """Generate a 384-dimensional normalized vector embedding for a single text.

        Checks the in-memory LRU cache first. On miss, encodes the text in a worker
        thread to prevent blocking the event loop.

        Args:
            text: Input string to embed.

        Returns:
            list[float]: 384-dimensional float vector.

        Raises:
            CacheError: If embedding generation fails.
        """
        cache_key = self._hash_text(text)
        cached = self._lru_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            vector = await asyncio.to_thread(self._encode_single_sync, text)
            self._lru_cache.set(cache_key, vector)
            return vector
        except Exception as e:
            if isinstance(e, CacheError):
                raise
            logger.error("local_embed_failed", text_len=len(text), error=str(e))
            raise CacheError(f"Local embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dimensional normalized vector embeddings for a list of texts.

        Leverages LRU cache for already seen texts and encodes uncached texts
        in a single batch via worker thread.

        Args:
            texts: List of input strings to embed.

        Returns:
            list[list[float]]: List of 384-dimensional float vectors matching input order.

        Raises:
            CacheError: If batch embedding generation fails.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        uncached_keys: list[str] = []

        for idx, text in enumerate(texts):
            key = self._hash_text(text)
            cached = self._lru_cache.get(key)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
                uncached_keys.append(key)

        if uncached_texts:
            try:
                encoded_batch = await asyncio.to_thread(
                    self._encode_batch_sync,
                    uncached_texts,
                )
                for orig_idx, key, vector in zip(
                    uncached_indices, uncached_keys, encoded_batch, strict=True
                ):
                    results[orig_idx] = vector
                    self._lru_cache.set(key, vector)
            except Exception as e:
                if isinstance(e, CacheError):
                    raise
                logger.error(
                    "local_batch_embed_failed",
                    batch_size=len(uncached_texts),
                    error=str(e),
                )
                raise CacheError(f"Local batch embedding failed: {e}") from e

        return [vec for vec in results if vec is not None]
