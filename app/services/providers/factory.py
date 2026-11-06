"""Factory module for creating and configuring LLM provider adapters.

Enables dynamic instantiation of supported provider adapters (mock, openai, anthropic, gemini)
based on application configuration and routing requests.
"""

import structlog

from app.core.config import Settings
from app.services.providers.anthropic import AnthropicProvider
from app.services.providers.base import BaseLLMProvider
from app.services.providers.gemini import GeminiProvider
from app.services.providers.mock import MockProvider
from app.services.providers.openai import OpenAIProvider

logger = structlog.get_logger(__name__)


class ProviderFactory:
    """Factory for instantiating and managing LLM provider adapter instances."""

    _PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {
        "mock": MockProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }

    @classmethod
    def create(cls, provider_name: str, settings: Settings) -> BaseLLMProvider:
        """Instantiate a single LLM provider adapter by name.

        Args:
            provider_name: Identifier of the provider ('mock', 'openai', 'anthropic', 'gemini').
            settings: Application configuration settings instance.

        Returns:
            BaseLLMProvider: Configured provider adapter instance.

        Raises:
            ValueError: If the requested provider_name is not supported.
        """
        canonical_name = provider_name.strip().lower()
        provider_cls = cls._PROVIDER_MAP.get(canonical_name)

        if not provider_cls:
            supported = ", ".join(sorted(cls._PROVIDER_MAP.keys()))
            logger.error(
                "Unsupported provider requested", provider=canonical_name, supported=supported
            )
            raise ValueError(
                f"Unsupported provider: '{provider_name}'. Supported providers: {supported}"
            )

        logger.info("Instantiating provider adapter", provider=canonical_name)

        if provider_cls is MockProvider:
            return MockProvider()
        return provider_cls(settings)  # type: ignore[call-arg]

    @classmethod
    def create_all(cls, settings: Settings) -> dict[str, BaseLLMProvider]:
        """Instantiate all supported LLM provider adapters.

        Args:
            settings: Application configuration settings instance.

        Returns:
            dict[str, BaseLLMProvider]: Mapping of provider names to configured adapter instances.
        """
        logger.info("Instantiating all supported LLM provider adapters")
        providers: dict[str, BaseLLMProvider] = {}
        for name in cls._PROVIDER_MAP:
            providers[name] = cls.create(name, settings)
        return providers

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """Return the list of all supported provider identifiers.

        Returns:
            list[str]: Sorted list of canonical provider names.
        """
        return sorted(cls._PROVIDER_MAP.keys())
