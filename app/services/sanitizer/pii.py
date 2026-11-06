"""PII Sanitization Service.

Provides regex-based detection and redaction of Personally Identifiable Information (PII),
including Social Security Numbers (SSN), credit card numbers, email addresses, phone numbers,
and exposed API tokens or bearer authentication headers.
"""

import copy
import re
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Standard raw regular expressions for PII detection
RAW_PII_PATTERNS: dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}\b",
    "API_KEY": r"\b(?:sk-|pk_|rk_|Bearer\s+)[A-Za-z0-9_\-]{20,}\b",
    "BEARER_TOKEN": r"Bearer\s+[A-Za-z0-9_\-\.]+",
}

# Module-level pre-compiled regex pattern dictionary
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    pii_type: re.compile(pattern, flags=re.IGNORECASE if pii_type in {"EMAIL"} else 0)
    for pii_type, pattern in RAW_PII_PATTERNS.items()
}


class PIISanitizer:
    """Sanitizer that identifies and redacts PII using pre-compiled regular expressions."""

    def __init__(
        self,
        custom_patterns: dict[str, str | re.Pattern[str]] | None = None,
    ) -> None:
        """Initialize the PIISanitizer and pre-compile all regex patterns.

        Args:
            custom_patterns: Optional additional or overriding PII patterns.
        """
        self._patterns: dict[str, re.Pattern[str]] = {}

        # Compile base patterns
        for pii_type, pattern in RAW_PII_PATTERNS.items():
            if isinstance(pattern, re.Pattern):
                self._patterns[pii_type] = pattern
            else:
                flags = re.IGNORECASE if pii_type in {"EMAIL"} else 0
                self._patterns[pii_type] = re.compile(pattern, flags=flags)

        # Merge custom patterns if provided
        if custom_patterns:
            for pii_type, pattern in custom_patterns.items():
                if isinstance(pattern, re.Pattern):
                    self._patterns[pii_type] = pattern
                else:
                    self._patterns[pii_type] = re.compile(pattern)

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact all detected PII in the given string with replacement placeholders.

        Iterates through all configured patterns in order and replaces matched
        substrings with `[REDACTED_{TYPE}]`. Logs detection events securely without
        revealing the underlying sensitive data.

        Args:
            text: Input string to scan and redact.

        Returns:
            tuple[str, list[str]]: A tuple containing the redacted text and a list
                of detected PII types (deduplicated, maintaining discovery order).
        """
        if not text:
            return text, []

        sanitized_text = text
        detected_types: list[str] = []

        for pii_type, pattern in self._patterns.items():
            if pattern.search(sanitized_text):
                detected_types.append(pii_type)
                sanitized_text = pattern.sub(f"[REDACTED_{pii_type}]", sanitized_text)

        if detected_types:
            logger.info(
                "PII detected and redacted from text",
                detected_types=detected_types,
                count=len(detected_types),
            )

        return sanitized_text, detected_types

    def redact_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Apply PII redaction to message payloads in chat completion format.

        Recursively processes the 'content' field of each message dictionary,
        handling both plain string content and structured multi-part content lists.

        Args:
            messages: List of message dictionaries (e.g. [{"role": "user", "content": "..."}]).

        Returns:
            tuple[list[dict[str, Any]], list[str]]: A tuple of sanitized message dictionaries
                and a list of all distinct PII types detected across all messages.
        """
        sanitized_messages: list[dict[str, Any]] = []
        all_detected_types: list[str] = []

        def _record_detected(types: list[str]) -> None:
            for t in types:
                if t not in all_detected_types:
                    all_detected_types.append(t)

        for message in messages:
            msg_copy = copy.deepcopy(message)
            content = msg_copy.get("content")

            if isinstance(content, str):
                redacted_content, detected = self.redact(content)
                msg_copy["content"] = redacted_content
                _record_detected(detected)
            elif isinstance(content, list):
                # Handle structured/multimodal content lists: [{"type": "text", "text": "..."}]
                sanitized_content_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                        part_copy = copy.deepcopy(part)
                        redacted_text, detected = self.redact(part["text"])
                        part_copy["text"] = redacted_text
                        sanitized_content_parts.append(part_copy)
                        _record_detected(detected)
                    elif isinstance(part, str):
                        redacted_part, detected = self.redact(part)
                        sanitized_content_parts.append(redacted_part)
                        _record_detected(detected)
                    else:
                        sanitized_content_parts.append(part)
                msg_copy["content"] = sanitized_content_parts

            sanitized_messages.append(msg_copy)

        return sanitized_messages, all_detected_types

    def contains_pii(self, text: str) -> bool:
        """Perform a fast check to determine if the text contains any PII.

        Stops on first match without performing string replacement or allocations.

        Args:
            text: Input string to inspect.

        Returns:
            bool: True if any configured PII pattern matches, False otherwise.
        """
        if not text:
            return False

        return any(pattern.search(text) for pattern in self._patterns.values())


__all__ = ["PIISanitizer", "PII_PATTERNS", "RAW_PII_PATTERNS"]
