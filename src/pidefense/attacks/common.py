# Authorized-testing-only: research evaluation of the project's own local demo.
"""Shared resource bounds for in-memory evaluation transforms."""

MAX_PAYLOAD_LENGTH = 16_384


def validate_payload(payload: str) -> None:
    """Bound input by Unicode code points; whitespace is valid, empty text is not."""
    if not isinstance(payload, str) or not payload:
        raise ValueError("payload must be a non-empty string")
    if len(payload) > MAX_PAYLOAD_LENGTH:
        raise ValueError("payload exceeds maximum length")
