# Authorized-testing-only: bounded research evaluation of the project's local detectors.
"""Small lexical substitutions and benign preambles for classifier fragility evaluation."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations

from numpy.random import Generator

from pidefense.attacks.common import MAX_PAYLOAD_LENGTH, validate_payload

SYNONYMS = {
    "ignore": "disregard",
    "previous": "prior",
    "instructions": "directions",
    "reveal": "disclose",
    "secret": "private",
    "tell": "inform",
    "show": "display",
}
PREAMBLES = ("For context, ", "Please help. ", "Background information: ")
MAX_SITES = 16
MAX_SUBSTITUTIONS = 3


@dataclass(frozen=True)
class Perturbation:
    """Budget counts substituted words plus words inserted by a preamble."""

    text: str = field(repr=False)
    budget: int
    method: str


def perturbations(
    payload: str, rng: Generator, *, max_budget: int = 8, max_candidates: int = 127
) -> Iterator[Perturbation]:
    """Yield nonidentity candidates by increasing word-edit budget, with bounded search.

    Randomize eligible positions once, retaining at most sixteen; replace at most
    three words and prepend at most one fixed preamble. Case/whitespace outside
    substitutions is preserved. Synonyms are context-dependent: no claim of exact
    semantic equivalence is made. This finite family is not an exhaustive attack.
    """
    validate_payload(payload)
    if type(max_budget) is not int or max_budget < 1:
        raise ValueError("max_budget must be a positive integer")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    sites = [m for m in re.finditer(r"\b\w+\b", payload) if m.group().lower() in SYNONYMS]
    sites = [sites[int(i)] for i in rng.permutation(len(sites))[:MAX_SITES]]
    prefixes = ["", *(PREAMBLES[int(i)] for i in rng.permutation(len(PREAMBLES)))]
    seen: set[str] = set()
    family_budget = MAX_SUBSTITUTIONS + max(len(p.split()) for p in PREAMBLES)
    for budget in range(1, min(max_budget, family_budget) + 1):
        for prefix in prefixes:
            changes = budget - len(prefix.split())
            if not 0 <= changes <= min(MAX_SUBSTITUTIONS, len(sites)):
                continue
            for selected in combinations(sites, changes):
                text = payload
                for match in sorted(selected, key=lambda m: m.start(), reverse=True):
                    replacement = SYNONYMS[match.group().lower()]
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group().istitle():
                        replacement = replacement.capitalize()
                    text = text[: match.start()] + replacement + text[match.end() :]
                text = prefix + text
                if text == payload or text in seen or len(text) > MAX_PAYLOAD_LENGTH:
                    continue
                seen.add(text)
                yield Perturbation(
                    text,
                    budget,
                    "preamble+synonym"
                    if prefix and changes
                    else "preamble"
                    if prefix
                    else "synonym",
                )
                if len(seen) >= max_candidates:
                    return
