"""SJTU LLM API wrapper for explanation generation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import requests
    from requests import RequestException
except ImportError:  # Keep fallback explanations usable before dependencies are installed.
    requests = None  # type: ignore[assignment]
    RequestException = OSError  # type: ignore[assignment]


class LLMAPIError(Exception):
    """Raised when the LLM API cannot return a usable response."""


def _load_dotenv_if_needed() -> None:
    """Load a local .env file without requiring python-dotenv."""

    if os.getenv("SJTU_LLM_API_URL") and os.getenv("SJTU_LLM_API_KEY"):
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


def _get_config() -> tuple[str, str, str]:
    _load_dotenv_if_needed()
    api_url = os.getenv("SJTU_LLM_API_URL", "").strip()
    api_key = os.getenv("SJTU_LLM_API_KEY", "").strip()
    model = os.getenv("SJTU_LLM_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    missing = [
        name
        for name, value in (
            ("SJTU_LLM_API_URL", api_url),
            ("SJTU_LLM_API_KEY", api_key),
            ("SJTU_LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise LLMAPIError("Missing LLM environment variables: " + ", ".join(missing))
    return api_url, api_key, model


def _candidate_urls(api_url: str) -> list[str]:
    """Accept either a full chat endpoint or a common API base URL."""

    base_url = api_url.rstrip("/")
    known_suffixes = ("/chat/completions", "/v1/chat/completions")
    if base_url.endswith(known_suffixes):
        return [base_url]
    return [
        base_url,
        f"{base_url}/v1/chat/completions",
        f"{base_url}/chat/completions",
    ]


def _extract_text(data: Any) -> str:
    """Extract generated text from OpenAI-compatible and common custom payloads."""

    if isinstance(data, str):
        return data.strip()

    if not isinstance(data, Mapping):
        raise LLMAPIError("Unexpected LLM response type")

    choices = data.get("choices")
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, Sequence):
                    parts = []
                    for item in content:
                        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                            parts.append(item["text"])
                        elif isinstance(item, str):
                            parts.append(item)
                    if parts:
                        return "".join(parts).strip()
            for key in ("text", "content", "answer", "response"):
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    for key in ("output_text", "text", "content", "answer", "response", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested_data = data.get("data")
    if isinstance(nested_data, Mapping):
        return _extract_text(nested_data)

    raise LLMAPIError("LLM response does not contain generated text")


def call_llm(
    messages: list[dict[str, str]],
    timeout: float = 30.0,
    retries: int = 2,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Call the SJTU LLM API and return generated text.

    Parameters follow OpenAI-compatible chat completion conventions.
    """

    if not isinstance(messages, list) or not messages:
        raise LLMAPIError("messages must be a non-empty list")
    if retries < 0:
        raise LLMAPIError("retries must be non-negative")
    if requests is None:
        raise LLMAPIError("The requests package is not installed")

    api_url, api_key, model = _get_config()
    candidate_urls = _candidate_urls(api_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        for url in candidate_urls:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=timeout,
                )
                if response.status_code >= 400:
                    raise LLMAPIError(
                        f"LLM API returned HTTP {response.status_code} at {url}: "
                        f"{response.text[:200]}"
                    )
                try:
                    data = response.json()
                except ValueError as exc:
                    raise LLMAPIError("LLM API returned non-JSON response") from exc
                return _extract_text(data)
            except (RequestException, LLMAPIError) as exc:
                last_error = exc
                if isinstance(exc, LLMAPIError) and "HTTP 404" in str(exc):
                    continue
                break
        if attempt < retries:
            time.sleep(min(2**attempt, 4))

    raise LLMAPIError(f"LLM API call failed after {retries + 1} attempts") from last_error


def call_sjtu_llm(
    prompt: str | list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 300,
    timeout: float = 20,
    retries: int = 2,
) -> str:
    """Call SJTU LLM with either a plain prompt string or chat messages."""

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt
    return call_llm(
        messages,
        timeout=timeout,
        retries=retries,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["LLMAPIError", "call_llm", "call_sjtu_llm"]
