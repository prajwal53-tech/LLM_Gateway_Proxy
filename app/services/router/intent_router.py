"""Intent-based router for dynamic model selection and upstream provider dispatching.

Analyzes incoming chat completion requests, routing headers, code patterns,
reasoning keywords, and payload sizes to intelligently route traffic across
Anthropic, OpenAI, and Google Gemini models.
"""

import re

from app.api.v1.schemas import ChatCompletionRequest
from app.core.logging import get_logger


class IntentRouter:
    """Intelligent router selecting the optimal LLM provider and model based on intent heuristics.

    Attributes:
        CODE_PATTERNS: List of regex patterns indicating programming or technical intent.
        COMPLEX_REASONING_KEYWORDS: List of keywords indicating deep reasoning or analytical intent.
        MODEL_MAP: Mapping of intent categories to default model identifiers.
    """

    CODE_PATTERNS: list[str] = [
        r"\bdef\s+",
        r"\bclass\s+",
        r"\bSELECT\s+",
        r"\bFROM\s+",
        r"\bINSERT\s+",
        r"```",
        r"\bfunction\s+",
        r"\bconst\s+",
        r"\bimport\s+",
        r"\breturn\s+",
    ]

    COMPLEX_REASONING_KEYWORDS: list[str] = [
        "explain",
        "analyze",
        "compare",
        "evaluate",
        "debug",
        "optimize",
        "architect",
        "design",
    ]

    MODEL_MAP: dict[str, str] = {
        "code_complex": "claude-3-5-sonnet-20241022",
        "long_context": "gpt-4o",
        "standard": "gemini-1.5-flash",
        "quality_first": "claude-3-5-sonnet-20241022",
        "cost_optimized": "gemini-1.5-flash",
    }

    def __init__(self) -> None:
        """Initialize the IntentRouter and configure structured logger."""
        self.logger = get_logger(__name__)

    def _detect_provider_from_model(self, model_name: str) -> str:
        """Map model name prefixes and identifiers to the corresponding upstream provider.

        Args:
            model_name: Target model identifier (e.g. 'gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash').

        Returns:
            str: Provider identifier ('openai', 'anthropic', 'gemini', or 'mock').
        """
        model_lower = model_name.lower().strip()
        if "/" in model_lower:
            prefix_provider, _, rest = model_lower.partition("/")
            if prefix_provider in ("openai", "anthropic", "gemini", "mock"):
                return prefix_provider
            model_lower = rest

        if (
            model_lower.startswith(("gpt", "o1", "o3", "o4", "text-", "dall-e", "chatgpt"))
            or "openai" in model_lower
        ):
            return "openai"
        if model_lower.startswith("claude") or "anthropic" in model_lower:
            return "anthropic"
        if model_lower.startswith("gemini") or "google" in model_lower:
            return "gemini"
        if model_lower.startswith("mock"):
            return "mock"
        return "openai"

    def _extract_last_user_content(self, request: ChatCompletionRequest) -> str:
        """Extract content from the last user message in the conversation.

        Args:
            request: Incoming chat completion request.

        Returns:
            str: The text content of the last user message, or last message content, or empty string.
        """
        if not request.messages:
            return ""

        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                return msg.content

        # Fallback to the last message if no explicit user role exists
        return request.messages[-1].content or ""

    def route(
        self,
        request: ChatCompletionRequest,
        headers: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Route a chat completion request to the optimal provider and model.

        Routing Evaluation Order:
        1. Explicit Model: If request.model is explicitly set, resolve provider and return.
        2. Routing Strategy Header: Check X-Routing-Strategy for 'quality-first' or 'cost-optimized'.
        3. Code Patterns: Check user message for code/programming syntax -> Claude 3.5 Sonnet.
        4. Complex Reasoning: Check user message for analytical keywords -> Claude 3.5 Sonnet.
        5. Long Context: Check if user message or context exceeds 2000 chars -> GPT-4o.
        6. Default: Fallback to cost-efficient standard -> Gemini 1.5 Flash.

        Args:
            request: The chat completion request payload.
            headers: Optional dictionary of HTTP request headers.

        Returns:
            tuple[str, str]: A tuple containing (provider_name, model_name).
        """
        headers = headers or {}
        user_content = self._extract_last_user_content(request)

        # 1. Explicit model requested
        if request.model:
            provider = request.provider or self._detect_provider_from_model(request.model)
            model = request.model
            reason = "explicit_model"
            self.logger.info(
                "request_routed",
                provider=provider,
                model=model,
                reason=reason,
                user_content_len=len(user_content),
            )
            return provider, model

        # 2. X-Routing-Strategy header check
        routing_strategy: str | None = None
        for key, value in headers.items():
            if key.lower() == "x-routing-strategy":
                routing_strategy = value.strip().lower()
                break

        if routing_strategy in ("quality-first", "quality_first"):
            provider = "anthropic"
            model = self.MODEL_MAP["quality_first"]
            reason = "header_quality_first"
            self.logger.info(
                "request_routed",
                provider=provider,
                model=model,
                reason=reason,
                strategy=routing_strategy,
            )
            return provider, model

        if routing_strategy in ("cost-optimized", "cost_optimized"):
            provider = "gemini"
            model = self.MODEL_MAP["cost_optimized"]
            reason = "header_cost_optimized"
            self.logger.info(
                "request_routed",
                provider=provider,
                model=model,
                reason=reason,
                strategy=routing_strategy,
            )
            return provider, model

        # 3. Check for code syntax patterns
        for pattern in self.CODE_PATTERNS:
            flags = (
                re.IGNORECASE if any(sql in pattern for sql in ("SELECT", "FROM", "INSERT")) else 0
            )
            if re.search(pattern, user_content, flags):
                provider = "anthropic"
                model = self.MODEL_MAP["code_complex"]
                reason = f"code_pattern_matched:{pattern}"
                self.logger.info(
                    "request_routed",
                    provider=provider,
                    model=model,
                    reason=reason,
                    user_content_len=len(user_content),
                )
                return provider, model

        # 4. Check for complex reasoning keywords
        for keyword in self.COMPLEX_REASONING_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", user_content, re.IGNORECASE):
                provider = "anthropic"
                model = self.MODEL_MAP["code_complex"]
                reason = f"complex_reasoning_keyword:{keyword}"
                self.logger.info(
                    "request_routed",
                    provider=provider,
                    model=model,
                    reason=reason,
                    keyword=keyword,
                )
                return provider, model

        # 5. Check token / character length (> 2000 chars)
        total_context_len = sum(len(m.content) for m in request.messages if m.content)
        if len(user_content) > 2000 or total_context_len > 2000:
            provider = "openai"
            model = self.MODEL_MAP["long_context"]
            reason = "long_context"
            self.logger.info(
                "request_routed",
                provider=provider,
                model=model,
                reason=reason,
                user_content_len=len(user_content),
                total_context_len=total_context_len,
            )
            return provider, model

        # 6. Default standard routing
        provider = "gemini"
        model = self.MODEL_MAP["standard"]
        reason = "standard_default"
        self.logger.info(
            "request_routed",
            provider=provider,
            model=model,
            reason=reason,
            user_content_len=len(user_content),
        )
        return provider, model
