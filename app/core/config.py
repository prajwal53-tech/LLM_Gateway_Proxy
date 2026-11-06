"""Application configuration module using Pydantic Settings.

Provides centralized typed settings loaded from environment variables and .env files,
with singleton access via get_settings().
"""

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core application settings and configuration.

    Attributes:
        APP_NAME: Name of the application service.
        APP_VERSION: Current service version.
        DEBUG: Debug mode flag.
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        AUTH_ENABLED: Whether API key authentication is enforced.
        GATEWAY_API_KEYS: Mapping of valid API keys to client/tenant identifiers.
        REDIS_URL: Redis Stack connection string URL.
        CACHE_TTL_SECONDS: Semantic cache entry time-to-live in seconds.
        CACHE_SIMILARITY_THRESHOLD: Maximum cosine distance threshold for cache hits.
        EMBEDDING_PROVIDER: Embedding provider implementation ("local", "openai", etc.).
        EMBEDDING_MODEL_NAME: Name of the embedding model to load.
        OPENAI_API_KEY: Optional OpenAI API key for upstream calls.
        ANTHROPIC_API_KEY: Optional Anthropic API key for upstream calls.
        GEMINI_API_KEY: Optional Google Gemini API key for upstream calls.
        DEFAULT_PROVIDER: Default LLM provider to route requests to ("mock", "openai", etc.).
        CB_FAILURE_THRESHOLD: Circuit breaker consecutive failure count before opening.
        CB_FAILURE_WINDOW_SECONDS: Time window in seconds to track circuit failures.
        CB_COOLDOWN_SECONDS: Duration in seconds before transitioning from OPEN to HALF-OPEN.
        CB_HALF_OPEN_PROBE_RATIO: Ratio of traffic permitted during HALF-OPEN probe state.
        CB_REQUEST_TIMEOUT_MS: Individual upstream request timeout in milliseconds.
        FALLBACK_CHAIN: Ordered list of provider names to try on circuit trip / failures.
    """

    # General Application Settings
    APP_NAME: str = "llm-eval-proxy"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Authentication & Security
    AUTH_ENABLED: bool = True
    GATEWAY_API_KEYS: dict[str, str] = Field(default_factory=lambda: {"dev-key-001": "dev_service"})

    # Redis & Semantic Caching
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 86400
    CACHE_SIMILARITY_THRESHOLD: float = 0.08

    # Embeddings
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # LLM Provider Credentials
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    DEFAULT_PROVIDER: str = "mock"

    # Circuit Breaker & Resilience
    CB_FAILURE_THRESHOLD: int = 5
    CB_FAILURE_WINDOW_SECONDS: int = 30
    CB_COOLDOWN_SECONDS: int = 60
    CB_HALF_OPEN_PROBE_RATIO: float = 0.10
    CB_REQUEST_TIMEOUT_MS: int = 8000

    # Failover & Routing
    FALLBACK_CHAIN: list[str] = Field(default_factory=lambda: ["gemini", "anthropic", "openai"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("GATEWAY_API_KEYS", mode="before")
    @classmethod
    def parse_gateway_api_keys(cls, value: Any) -> dict[str, str]:
        """Parse GATEWAY_API_KEYS from JSON string if passed as environment variable."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                # Comma-separated key:value fallback format: "k1:v1,k2:v2"
                key_dict = {}
                for pair in value.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        key_dict[k.strip()] = v.strip()
                if key_dict:
                    return key_dict
        elif isinstance(value, dict):
            return value
        return {"dev-key-001": "dev_service"}

    @field_validator("FALLBACK_CHAIN", mode="before")
    @classmethod
    def parse_fallback_chain(cls, value: Any) -> list[str]:
        """Parse FALLBACK_CHAIN from JSON array or comma-delimited string."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
            except json.JSONDecodeError:
                return [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            return [str(item).strip() for item in value]
        return ["gemini", "anthropic", "openai"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve singleton cached Settings instance.

    Returns:
        Settings: Application configuration settings instance.
    """
    return Settings()
