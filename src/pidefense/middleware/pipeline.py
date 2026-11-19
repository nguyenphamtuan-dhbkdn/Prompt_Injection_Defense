"""Compose input, instruction-separation and output defenses."""

import logging
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal, Protocol

from pidefense.demo.ollama_client import OllamaClient
from pidefense.middleware.classifier_guard import ClassifierGuard, GuardDecision
from pidefense.middleware.output_filter import check_response_for_leak_or_compliance
from pidefense.middleware.system_prompt import SYSTEM_PROMPT, wrap_untrusted

_LOGGER = logging.getLogger(__name__)


class ChatClient(Protocol):
    """Transport contract for local client and offline integration doubles."""

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Return assistant text without logging arguments or response."""
        ...


@dataclass(frozen=True)
class ChatTurnResult:
    """Only accepted text is exposed; blocked/error results contain no output."""

    status: Literal["allowed", "blocked", "error"]
    response: str | None = field(default=None, repr=False)
    decision: GuardDecision | None = None
    rule: str = "classifier"


def _audit(text: str, decision: GuardDecision, rule: str) -> None:
    """Log only length, truncated SHA-256, label, confidence and fixed rule ID."""
    fingerprint = sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()[:8]
    _LOGGER.info(
        "length=%d fingerprint=%s label=%s score=%.6f rule=%s",
        len(text),
        fingerprint,
        decision.label,
        decision.score,
        rule,
    )


class GuardedChat:
    """Screen every untrusted source before a stateless local chat request."""

    def __init__(self, guard: ClassifierGuard, client: ChatClient | None = None) -> None:
        """Use an injected local transport or the loopback Ollama default."""
        self.guard = guard
        self.client = client if client is not None else OllamaClient()

    def send(self, user_message: str, context_document: str | None = None) -> ChatTurnResult:
        """Fail closed on layer errors and withhold flagged assistant responses."""
        try:
            content = wrap_untrusted(user_message, context_document)
            decisions: list[GuardDecision] = []
            for text in (user_message, context_document):
                if text is None:
                    continue
                decision = self.guard.check(text)
                _audit(text, decision, "classifier")
                decisions.append(decision)
                if decision.label == "injection":
                    return ChatTurnResult("blocked", decision=decision)
            decision = max(decisions, key=lambda item: item.score)
            response = self.client.chat(SYSTEM_PROMPT, content)
            if check_response_for_leak_or_compliance(response, SYSTEM_PROMPT):
                _audit(response, GuardDecision("injection", 1.0), "output_filter")
                return ChatTurnResult("blocked", decision=decision, rule="output_filter")
            return ChatTurnResult("allowed", response, decision, "passed")
        except Exception:
            # Backend exceptions may contain raw text. Do not log or retain them.
            return ChatTurnResult("error", rule="layer_error")
