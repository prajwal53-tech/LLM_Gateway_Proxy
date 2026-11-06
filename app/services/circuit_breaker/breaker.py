"""Circuit breaker resilience implementation for upstream LLM provider calls.

Provides sliding-window failure tracking, state transitions (CLOSED, OPEN, HALF_OPEN),
probe traffic throttling, and timeout enforcement.
"""

import asyncio
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

from app.core.config import Settings, get_settings
from app.core.exceptions import CircuitOpenError, ProviderAPIError, RateLimitError
from app.core.logging import get_logger

T = TypeVar("T")


class CircuitState(StrEnum):
    """Enumeration of possible circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker protecting upstream providers against cascading failures.

    Attributes:
        provider_name: Identifier for the protected LLM provider.
        settings: Application settings containing failure thresholds and timeouts.
        state: Current operational state (CLOSED, OPEN, HALF_OPEN).
        failure_timestamps: Sliding-window timestamps of observed provider failures.
        last_failure_time: Timestamp of the most recent failure occurrence.
    """

    def __init__(self, provider_name: str, settings: Settings | None = None) -> None:
        """Initialize the circuit breaker for a specific provider.

        Args:
            provider_name: Identifier of the provider (e.g. 'openai', 'anthropic', 'gemini').
            settings: Optional Settings instance. Defaults to get_settings().
        """
        self.provider_name = provider_name
        self.settings: Settings = settings or get_settings()
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_timestamps: deque[float] = deque()
        self.last_failure_time: float | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self.logger = get_logger(__name__)

    @property
    def state_value(self) -> int:
        """Numeric representation of current circuit state for metrics export.

        Mapping:
            0 = CLOSED
            1 = OPEN
            2 = HALF_OPEN

        Returns:
            int: State numeric code.
        """
        if self.state == CircuitState.CLOSED:
            return 0
        if self.state == CircuitState.OPEN:
            return 1
        if self.state == CircuitState.HALF_OPEN:
            return 2
        return 0

    def _cooldown_elapsed(self) -> bool:
        """Check if the configured cooldown duration has elapsed since last failure.

        Returns:
            bool: True if cooldown period has passed, False otherwise.
        """
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.settings.CB_COOLDOWN_SECONDS

    async def can_execute(self) -> bool:
        """Determine whether a request is allowed to execute through the circuit breaker.

        State transition logic:
        - CLOSED: Always returns True.
        - OPEN: Checks if cooldown period has elapsed. If so, transitions to HALF_OPEN
          and permits probe traffic based on CB_HALF_OPEN_PROBE_RATIO. If cooldown has
          not elapsed, returns False.
        - HALF_OPEN: Allows probe traffic with probability CB_HALF_OPEN_PROBE_RATIO.

        Returns:
            bool: True if request can proceed; False if circuit is OPEN.
        """
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if self._cooldown_elapsed():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info(
                        "circuit_state_transition",
                        provider_name=self.provider_name,
                        from_state=CircuitState.OPEN.value,
                        to_state=CircuitState.HALF_OPEN.value,
                        reason="cooldown_elapsed",
                    )
                    return random.random() < self.settings.CB_HALF_OPEN_PROBE_RATIO
                return False

            if self.state == CircuitState.HALF_OPEN:
                return random.random() < self.settings.CB_HALF_OPEN_PROBE_RATIO

            return False

    async def _on_success(self) -> None:
        """Handle a successful request execution.

        If in HALF_OPEN state, transitions back to CLOSED and resets tracked failures.
        """
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                old_state = self.state
                self.state = CircuitState.CLOSED
                self.failure_timestamps.clear()
                self.last_failure_time = None
                self.logger.info(
                    "circuit_state_transition",
                    provider_name=self.provider_name,
                    from_state=old_state.value,
                    to_state=CircuitState.CLOSED.value,
                    reason="probe_success",
                )
            elif self.state == CircuitState.CLOSED:
                now = time.time()
                window = self.settings.CB_FAILURE_WINDOW_SECONDS
                while self.failure_timestamps and self.failure_timestamps[0] < now - window:
                    self.failure_timestamps.popleft()

    async def _on_failure(self) -> None:
        """Record a failure and evaluate circuit breaker tripping conditions."""
        async with self._lock:
            now = time.time()
            self.last_failure_time = now
            self.failure_timestamps.append(now)

            # Prune failures outside the sliding window
            window = self.settings.CB_FAILURE_WINDOW_SECONDS
            while self.failure_timestamps and self.failure_timestamps[0] < now - window:
                self.failure_timestamps.popleft()

            failures_in_window = len(self.failure_timestamps)
            old_state = self.state

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.logger.warning(
                    "circuit_state_transition",
                    provider_name=self.provider_name,
                    from_state=old_state.value,
                    to_state=CircuitState.OPEN.value,
                    reason="half_open_probe_failed",
                    failure_count=failures_in_window,
                )
            elif self.state == CircuitState.CLOSED:
                if failures_in_window >= self.settings.CB_FAILURE_THRESHOLD:
                    self.state = CircuitState.OPEN
                    self.logger.warning(
                        "circuit_state_transition",
                        provider_name=self.provider_name,
                        from_state=old_state.value,
                        to_state=CircuitState.OPEN.value,
                        reason="failure_threshold_exceeded",
                        failure_count=failures_in_window,
                        threshold=self.settings.CB_FAILURE_THRESHOLD,
                        window_seconds=window,
                    )

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an asynchronous operation protected by the circuit breaker.

        Args:
            func: Asynchronous callable to execute.
            *args: Positional arguments to forward to func.
            **kwargs: Keyword arguments to forward to func.

        Returns:
            T: Result from the executed callable.

        Raises:
            CircuitOpenError: If the circuit is open and request is not permitted.
            RateLimitError: If upstream rate limit is hit.
            ProviderAPIError: If upstream provider returns an error.
            TimeoutError: If execution exceeds configured timeout.
            Exception: Any other unhandled exception raised by func.
        """
        if not await self.can_execute():
            raise CircuitOpenError(provider_name=self.provider_name)

        timeout_seconds = self.settings.CB_REQUEST_TIMEOUT_MS / 1000.0
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
        except (RateLimitError, ProviderAPIError, TimeoutError):
            await self._on_failure()
            raise
        except Exception:
            raise

        await self._on_success()
        return result


class CircuitBreakerRegistry:
    """Thread-safe registry managing CircuitBreaker instances per provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the registry.

        Args:
            settings: Application settings. If None, loaded via get_settings().
        """
        self.settings: Settings = settings or get_settings()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider_name: str) -> CircuitBreaker:
        """Get existing CircuitBreaker for provider or create a new one.

        Args:
            provider_name: Identifier of the provider (e.g. 'openai', 'anthropic', 'gemini').

        Returns:
            CircuitBreaker: The circuit breaker instance for the provider.
        """
        if provider_name not in self._breakers:
            self._breakers[provider_name] = CircuitBreaker(
                provider_name=provider_name,
                settings=self.settings,
            )
        return self._breakers[provider_name]

    def get_all(self) -> dict[str, CircuitBreaker]:
        """Return all registered circuit breakers.

        Returns:
            dict[str, CircuitBreaker]: Mapping of provider name to circuit breaker.
        """
        return dict(self._breakers)
