"""Factory for creating and managing embedding service singleton instances.

Maps provider names ('local', 'openai', 'gemini') to their respective
implementations and returns singleton instances.
"""

import threading
from typing import Any

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.embeddings.base import BaseEmbeddingService
from app.services.embeddings.gemini_embed import GeminiEmbeddingService
from app.services.embeddings.local import LocalEmbeddingService
from app.services.embeddings.openai_embed import OpenAIEmbeddingService

logger = get_logger(__name__)


class EmbeddingServiceFactory:
    """Factory for embedding services with thread-safe singleton caching."""

    _instances: dict[str, BaseEmbeddingService] = {}
    _lock = threading.Lock()

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingService:
        """Create or retrieve a singleton embedding service instance.

        Args:
            settings: Application settings containing EMBEDDING_PROVIDER and model config.
            provider: Optional explicit provider override ('local', 'openai', 'gemini').
            model_name: Optional explicit embedding model name override.
            **kwargs: Additional parameters forwarded to service constructor.

        Returns:
            BaseEmbeddingService: Configured embedding service instance.

        Raises:
            ValueError: If an unsupported provider name is specified.
        """
        app_settings = settings or get_settings()
        target_provider = (provider or app_settings.EMBEDDING_PROVIDER).lower().strip()
        target_model = model_name or app_settings.EMBEDDING_MODEL_NAME

        cache_key = f"{target_provider}:{target_model}"

        with cls._lock:
            if cache_key in cls._instances:
                return cls._instances[cache_key]

            service: BaseEmbeddingService
            if target_provider == "local":
                service = LocalEmbeddingService(
                    model_name=target_model,
                    **kwargs,
                )
            elif target_provider == "openai":
                api_key = kwargs.pop("api_key", None) or app_settings.OPENAI_API_KEY
                service = OpenAIEmbeddingService(
                    api_key=api_key,
                    model_name=target_model,
                    **kwargs,
                )
            elif target_provider == "gemini":
                api_key = kwargs.pop("api_key", None) or app_settings.GEMINI_API_KEY
                service = GeminiEmbeddingService(
                    api_key=api_key,
                    model_name=target_model,
                    **kwargs,
                )
            else:
                supported = ["local", "openai", "gemini"]
                logger.error(
                    "unsupported_embedding_provider",
                    provider=target_provider,
                    supported=supported,
                )
                raise ValueError(
                    f"Unsupported embedding provider '{target_provider}'. "
                    f"Must be one of: {', '.join(supported)}"
                )

            cls._instances[cache_key] = service
            logger.info(
                "embedding_service_created",
                provider=target_provider,
                model=target_model,
                dimensions=service.dimensions,
            )
            return service

    @classmethod
    def reset(cls) -> None:
        """Clear all cached singleton service instances.

        Useful for unit testing and reconfiguration.
        """
        with cls._lock:
            cls._instances.clear()
            logger.debug("embedding_service_factory_reset")
