"""Unit tests for the PIISanitizer service.

Verifies regex pattern detection and redaction of Social Security Numbers,
credit card numbers, email addresses, phone numbers, API keys, and nested chat messages.
"""

from typing import Any

from app.services.sanitizer.pii import PIISanitizer


def test_redact_ssn(pii_sanitizer: PIISanitizer) -> None:
    """Test Social Security Number (SSN) pattern detection and redaction."""
    text = "My SSN is 123-45-6789"
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == "My SSN is [REDACTED_SSN]"
    assert "SSN" in detected


def test_redact_credit_card(pii_sanitizer: PIISanitizer) -> None:
    """Test credit card number pattern detection and redaction."""
    text = "Card: 4111-1111-1111-1111"
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == "Card: [REDACTED_CREDIT_CARD]"
    assert "CREDIT_CARD" in detected


def test_redact_email(pii_sanitizer: PIISanitizer) -> None:
    """Test email address pattern detection and redaction."""
    text = "Email me at test@example.com"
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == "Email me at [REDACTED_EMAIL]"
    assert "EMAIL" in detected


def test_redact_phone(pii_sanitizer: PIISanitizer) -> None:
    """Test phone number pattern detection and redaction."""
    text = "Call 555-123-4567"
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == "Call [REDACTED_PHONE]"
    assert "PHONE" in detected


def test_redact_api_key(pii_sanitizer: PIISanitizer) -> None:
    """Test exposed API key / bearer token detection and redaction."""
    text = "Key: sk-abc123def456ghi789jkl012"
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == "Key: [REDACTED_API_KEY]"
    assert "API_KEY" in detected


def test_no_pii_unchanged(pii_sanitizer: PIISanitizer) -> None:
    """Test that text without PII is returned completely unaltered."""
    text = "Explain the difference between synchronous and asynchronous execution in Python."
    redacted_text, detected = pii_sanitizer.redact(text)

    assert redacted_text == text
    assert len(detected) == 0


def test_multiple_pii_types(pii_sanitizer: PIISanitizer) -> None:
    """Test that text containing multiple distinct PII types has all sensitive fields redacted."""
    text = (
        "User contact info: SSN 123-45-6789, email alice@domain.org, "
        "phone (555) 234-5678, card 4111-2222-3333-4444, and key sk-supersecretkey1234567890"
    )
    redacted_text, detected = pii_sanitizer.redact(text)

    assert "[REDACTED_SSN]" in redacted_text
    assert "[REDACTED_EMAIL]" in redacted_text
    assert "[REDACTED_PHONE]" in redacted_text
    assert "[REDACTED_CREDIT_CARD]" in redacted_text
    assert "[REDACTED_API_KEY]" in redacted_text
    assert set(detected) == {"SSN", "EMAIL", "PHONE", "CREDIT_CARD", "API_KEY"}


def test_contains_pii(pii_sanitizer: PIISanitizer) -> None:
    """Test rapid boolean PII detection without allocations."""
    assert pii_sanitizer.contains_pii("Contact me at user@test.com") is True
    assert pii_sanitizer.contains_pii("SSN: 987-65-4321") is True
    assert pii_sanitizer.contains_pii("Call 555-444-3333") is True
    assert pii_sanitizer.contains_pii("Clean text with no sensitive data whatsoever.") is False
    assert pii_sanitizer.contains_pii("") is False


def test_redact_messages(pii_sanitizer: PIISanitizer) -> None:
    """Test message list redaction for OpenAI chat structure."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My email is bob@corp.com and my SSN is 111-22-3333."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Also my phone is 555-888-9999."},
                {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
            ],
        },
    ]

    sanitized_messages, all_detected = pii_sanitizer.redact_messages(messages)

    assert len(sanitized_messages) == 3
    # System message unchanged
    assert sanitized_messages[0]["content"] == "You are a helpful assistant."

    # User message 1 redacted
    assert "[REDACTED_EMAIL]" in sanitized_messages[1]["content"]
    assert "[REDACTED_SSN]" in sanitized_messages[1]["content"]

    # Structured multipart content redacted
    assert isinstance(sanitized_messages[2]["content"], list)
    assert "[REDACTED_PHONE]" in sanitized_messages[2]["content"][0]["text"]

    assert "EMAIL" in all_detected
    assert "SSN" in all_detected
    assert "PHONE" in all_detected
