# Authorized-testing-only: research evaluation of the project's own local demo.
"""Lossless splitting into the simple two/three-turn scenario specified for M4."""

from pidefense.attacks.common import validate_payload


def instruction_splitting(payload: str, n_turns: int = 2) -> list[str]:
    """Partition code points into balanced contiguous chunks; concatenation is exact.

    Require enough code points for non-empty turns. Splitting does not establish
    that individual turns are benign or that a chatbot will reconstruct intent.
    """
    validate_payload(payload)
    if type(n_turns) is not int or n_turns not in (2, 3):
        raise ValueError("n_turns must be integer 2 or 3")
    if len(payload) < n_turns:
        raise ValueError("payload must contain at least n_turns code points")
    return [
        payload[i * len(payload) // n_turns : (i + 1) * len(payload) // n_turns]
        for i in range(n_turns)
    ]
