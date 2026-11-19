# Authorized-testing-only: research evaluation of the project's own local demo.
"""Pass through labeled source rows without authoring new direct instructions."""

from dataclasses import replace

from pidefense.attacks.common import validate_payload
from pidefense.data.schema import PromptSample


def direct_injection(sample: PromptSample) -> PromptSample:
    """Preserve source text and provenance, requiring an injection-labeled row."""
    validate_payload(sample.text)
    if sample.label != 1:
        raise ValueError("direct injection requires an injection-labeled source row")
    return replace(sample, technique="direct")
