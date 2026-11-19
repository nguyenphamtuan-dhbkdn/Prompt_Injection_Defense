"""Nonstreaming local Ollama /api/chat transport without payload logging."""

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Literal, TypedDict
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pidefense.config import MiddlewareConfig


class ChatMessage(TypedDict):
    """One role/content entry in Ollama's explicit conversation history."""

    role: Literal["system", "user", "assistant"]
    content: str


class OllamaError(RuntimeError):
    """Report a fixed transport/protocol failure without disclosing server text."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        """Reject redirects so a loopback response cannot forward the payload."""
        return None


class OllamaClient:
    """Only call configured loopback endpoints; disable proxies and redirects."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "phi3:mini",
        timeout: float = 60.0,
    ) -> None:
        """Validate destination with the existing middleware configuration."""
        cfg = MiddlewareConfig(ollama_host=host, ollama_model=model)
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("Ollama timeout must be finite and positive")
        self.host = cfg.ollama_host.rstrip("/")
        self.model = cfg.ollama_model
        self.timeout = timeout
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Send a stateless conversation using the same validated HTTP transport."""
        return self.chat_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )

    def chat_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        options: Mapping[str, int | float] | None = None,
    ) -> str:
        """POST role/content messages with stream=false; read message.content.

        Contract: https://docs.ollama.com/api/chat. Explicit history supports
        the research harness's bounded instruction-splitting conversations.
        Generation options are documented at https://docs.ollama.com/modelfile.
        Bound response bytes; never propagate HTTP bodies or exception text.
        """
        if not messages or any(
            m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
            for m in messages
        ):
            raise OllamaError("Invalid local conversation")
        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "messages": list(messages),
        }
        if options is not None:
            payload["options"] = dict(options)
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self.host + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as reply:
                raw = reply.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("Response too large")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("done") is not True:
                raise ValueError("Incomplete response")
            message = data.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("Invalid message")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Invalid content")
        except (URLError, OSError, ValueError, TypeError):
            raise OllamaError("Local Ollama unavailable or invalid response") from None
        return content
