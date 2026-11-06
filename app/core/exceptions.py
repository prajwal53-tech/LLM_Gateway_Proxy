"""Custom exception hierarchy for the LLM gateway and proxy service.

All domain-specific exceptions inherit from GatewayError to allow clean
error handling across middleware, services, and API layers.
"""

from typing import Any


class GatewayError(Exception):
    """Base exception for all LLM Gateway errors.

    Attributes:
        message: Human-readable error description.
        status_code: Suggested HTTP status code for response.
        error_code: String code for machine-readable error identification.
        details: Optional additional structured context.
    """

    def __init__(
        self,
        message: str = "An internal gateway error occurred.",
        status_code: int = 500,
        error_code: str = "INTERNAL_GATEWAY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] (Status {self.status_code}): {self.message}"


class ProviderAPIError(GatewayError):
    """Exception raised when an upstream LLM provider API call fails.

    Attributes:
        provider_name: The identifier of the failing provider (e.g. 'openai', 'gemini').
        status_code: Upstream or mapped HTTP status code.
        message: Detailed error message from the provider.
        details: Raw error response payload if available.
    """

    def __init__(
        self,
        provider_name: str,
        status_code: int = 502,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        msg = message or f"Upstream error from provider '{provider_name}'."
        super().__init__(
            message=msg,
            status_code=status_code,
            error_code="PROVIDER_API_ERROR",
            details={"provider_name": provider_name, **(details or {})},
        )


class RateLimitError(ProviderAPIError):
    """Exception raised when an upstream provider or internal rate limiter triggers a 429.

    Attributes:
        provider_name: The identifier of the provider that rate limited the request.
        retry_after: Optional number of seconds after which the client should retry.
        message: Human-readable message.
    """

    def __init__(
        self,
        provider_name: str,
        retry_after: float | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.retry_after = retry_after
        err_details = {"provider_name": provider_name, "retry_after": retry_after}
        if details:
            err_details.update(details)
        msg = message or f"Rate limit exceeded for provider '{provider_name}'."
        super().__init__(
            provider_name=provider_name,
            status_code=429,
            message=msg,
            details=err_details,
        )
        self.error_code = "RATE_LIMIT_EXCEEDED"


class CircuitOpenError(GatewayError):
    """Exception raised when a request is rejected because the provider's circuit breaker is open.

    Attributes:
        provider_name: The provider whose circuit is currently OPEN.
        message: Explanation of the circuit state.
    """

    def __init__(
        self,
        provider_name: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        msg = (
            message
            or f"Circuit breaker for provider '{provider_name}' is OPEN. Requests temporarily halted."
        )
        super().__init__(
            message=msg,
            status_code=503,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={"provider_name": provider_name, **(details or {})},
        )


class AllProvidersFailedError(GatewayError):
    """Exception raised when all providers in the fallback chain have failed or are unavailable.

    Attributes:
        message: Summary of failures across the chain.
        details: Dict of errors collected from each attempted provider.
    """

    def __init__(
        self,
        message: str = "All configured LLM providers in the fallback chain failed.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="ALL_PROVIDERS_FAILED",
            details=details or {},
        )


class CacheError(GatewayError):
    """Exception raised when semantic cache read, write, or indexing fails.

    Attributes:
        message: Description of cache failure.
        details: Contextual details regarding the operation.
    """

    def __init__(
        self,
        message: str = "Semantic cache operation failed.",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="CACHE_ERROR",
            details=details or {},
        )


class AuthenticationError(GatewayError):
    """Exception raised when authentication fails (missing or invalid API key).

    Attributes:
        message: Reason for authentication failure.
    """

    def __init__(
        self,
        message: str = "Invalid or missing API key.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_FAILED",
            details=details or {},
        )


class PIIDetectionError(GatewayError):
    """Exception raised when PII detection or redaction policy triggers an error.

    Attributes:
        message: Description of PII violation or failure.
        detected_types: List of PII types detected in the payload.
    """

    def __init__(
        self,
        message: str = "PII detected in payload violating strict policy.",
        detected_types: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.detected_types = detected_types or []
        err_details = {"detected_types": self.detected_types}
        if details:
            err_details.update(details)
        super().__init__(
            message=message,
            status_code=400,
            error_code="PII_POLICY_VIOLATION",
            details=err_details,
        )
