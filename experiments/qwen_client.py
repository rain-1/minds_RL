"""Hugging Face client utilities for querying Qwen models."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

import requests

ChatMessage = Mapping[str, Any]


def _normalize_message(message: ChatMessage) -> MutableMapping[str, Any]:
    """Return a shallow copy of a chat message suitable for inference APIs."""

    if not isinstance(message, Mapping):  # pragma: no cover - defensive
        raise TypeError(f"Expected mapping for chat message, received {type(message)!r}")
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise TypeError("Chat messages must contain string `role` and `content` fields")
    normalized: MutableMapping[str, Any] = {"role": role, "content": content}
    for key, value in message.items():
        if key in {"role", "content"}:
            continue
        normalized[key] = value
    return normalized


@dataclass(slots=True)
class QwenChatCompletionClient:
    """Small helper around Hugging Face's OpenAI-compatible inference API."""

    model_id: str = "Qwen/Qwen3-14B"
    base_url: str = "https://api-inference.huggingface.co/v1"
    token: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 2.0
    _session: requests.Session = field(default_factory=requests.Session, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.token is None:
            self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    def _headers(self) -> Mapping[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def chat_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        extra_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Call the OpenAI-compatible chat completions endpoint for Qwen."""

        if not messages:
            raise ValueError("At least one message is required to request a completion")
        normalized_messages = [_normalize_message(message) for message in messages]
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": normalized_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(dict(extra_body))

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 503} and attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, Mapping):
                    raise ValueError("Unexpected response payload from Hugging Face API")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        assert last_error is not None  # pragma: no cover - loop ensures value
        raise last_error

__all__ = ["QwenChatCompletionClient"]
