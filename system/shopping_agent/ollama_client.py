"""Shared, local-only Ollama chat transport for every active LLM role."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from threading import Lock
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

try:
    from .model_client import ModelCall, ModelError
except ImportError:
    from model_client import ModelCall, ModelError


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30.0
DEFAULT_OLLAMA_RETRIES = 1

Transport = Callable[[str, bytes, float], object]


class OllamaError(ModelError):
    """Base class for exhausted Ollama calls."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        latency_seconds: float,
        attempts: int,
        role: str,
        cause_type: str,
    ) -> None:
        super().__init__(message, model=model, latency_seconds=latency_seconds,
                         attempts=attempts, role=role, cause_type=cause_type,
                         provider="ollama")


class OllamaConfigurationError(OllamaError):
    """Raised when local model settings are invalid or conflict with legacy settings."""


class OllamaRequestError(OllamaError):
    """Raised when all transport attempts fail."""


class OllamaTimeoutError(OllamaRequestError):
    """Raised when the final failed transport attempt is a timeout."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama repeatedly returns an invalid response."""


@dataclass(frozen=True)
class OllamaCall(ModelCall):
    def instrumentation(self) -> dict[str, Any]:
        # Preserve the historical Ollama telemetry shape for compatibility.
        return {
            "role": self.role, "model": self.model,
            "latency_seconds": self.latency_seconds, "attempts": self.attempts,
            "retry_count": self.retry_count, "success": True,
            "error_type": None, "cause_type": None,
        }


def _environment_timeout() -> float:
    raw = os.environ.get("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be numeric") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than zero")
    return timeout


def _environment_host() -> str:
    host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip().rstrip("/")
    if not host:
        raise ValueError("OLLAMA_HOST must be non-empty")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def _configured_model(explicit_model: str | None) -> str:
    model = str(explicit_model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip()
    if not model:
        raise ValueError("OLLAMA_MODEL must be non-empty")
    legacy = os.environ.get("WINSTON_PARSER_MODEL")
    if legacy is not None and legacy.strip() and legacy.strip() != model:
        raise ValueError(
            "WINSTON_PARSER_MODEL is no longer supported and conflicts with OLLAMA_MODEL; "
            "remove WINSTON_PARSER_MODEL and set OLLAMA_MODEL for every active LLM role"
        )
    return model


class OllamaClient:
    """Small Ollama `/api/chat` client with one retry and structured telemetry."""

    provider = "ollama"

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        retries: int = DEFAULT_OLLAMA_RETRIES,
        transport: Transport | None = None,
    ) -> None:
        try:
            self.model = _configured_model(model)
            self.host = str(host or _environment_host()).strip().rstrip("/")
            if not self.host.startswith(("http://", "https://")):
                self.host = "http://" + self.host
            self.timeout_seconds = (
                _environment_timeout() if timeout_seconds is None else float(timeout_seconds)
            )
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be greater than zero")
            if isinstance(retries, bool) or int(retries) != retries or retries < 0:
                raise ValueError("retries must be a non-negative integer")
            self.retries = int(retries)
        except ValueError as exc:
            raise OllamaConfigurationError(
                str(exc),
                model=str(model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)),
                latency_seconds=0.0,
                attempts=0,
                role="configuration",
                cause_type=type(exc).__name__,
            ) from exc
        self._transport = transport or self._urlopen_transport
        self._calls: list[dict[str, Any]] = []
        self._lock = Lock()

    @staticmethod
    def _urlopen_transport(url: str, body: bytes, timeout_seconds: float) -> bytes:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()

    @staticmethod
    def _decode_response(response: object) -> tuple[str, str | None]:
        if isinstance(response, bytes):
            response = response.decode("utf-8")
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, Mapping):
            raise ValueError("Ollama response must be a JSON object")
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Ollama response is missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama response message content must be non-empty text")
        actual_model = response.get("model")
        if actual_model is not None and not isinstance(actual_model, str):
            raise ValueError("Ollama response model must be text when present")
        return content.strip(), actual_model

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, urllib.error.URLError)):
            if isinstance(exc, urllib.error.URLError) and not isinstance(exc.reason, TimeoutError):
                return False
            return True
        return "timed out" in str(exc).lower()

    def _record(self, trace: dict[str, Any]) -> None:
        with self._lock:
            self._calls.append(deepcopy(trace))

    def chat_result(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        format: str | Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        role: str = "chat",
        validator: Callable[[str], object] | None = None,
    ) -> OllamaCall:
        normalized_messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
        ]
        if not normalized_messages:
            raise ValueError("messages must be non-empty")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
        }
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = dict(options)
        body = json.dumps(payload).encode("utf-8")

        started = time.perf_counter()
        attempts = self.retries + 1
        last_error: Exception | None = None
        last_was_response_error = False
        for attempt in range(1, attempts + 1):
            try:
                response = self._transport(
                    f"{self.host}/api/chat", body, self.timeout_seconds
                )
                content, actual_model = self._decode_response(response)
                if validator is not None:
                    validator(content)
                elif format is not None:
                    json.loads(content)
                call = OllamaCall(
                    content=content,
                    model=str(actual_model or self.model),
                    latency_seconds=time.perf_counter() - started,
                    attempts=attempt,
                    role=role,
                    provider=self.provider,
                )
                self._record(call.instrumentation())
                return call
            except Exception as exc:  # transport and invalid responses share retry policy
                last_error = exc
                last_was_response_error = isinstance(
                    exc, (ValueError, TypeError, KeyError, json.JSONDecodeError)
                )

        latency = time.perf_counter() - started
        error_class: type[OllamaError]
        if last_was_response_error:
            error_class = OllamaResponseError
        elif last_error is not None and self._is_timeout(last_error):
            error_class = OllamaTimeoutError
        else:
            error_class = OllamaRequestError
        error = error_class(
            f"Ollama {role} call failed after {attempts} attempts: {last_error}",
            model=self.model,
            latency_seconds=latency,
            attempts=attempts,
            role=role,
            cause_type=type(last_error).__name__ if last_error is not None else "UnknownError",
        )
        self._record(error.instrumentation())
        raise error from last_error

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        format: str | Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        role: str = "chat",
        validator: Callable[[str], object] | None = None,
    ) -> str:
        return self.chat_result(
            messages,
            format=format,
            options=options,
            role=role,
            validator=validator,
        ).content

    def instrumentation(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._calls)


_DEFAULT_CLIENT: OllamaClient | None = None
_DEFAULT_CLIENT_LOCK = Lock()


def get_default_ollama_client() -> OllamaClient:
    """Return the process-wide client used by active local LLM roles."""

    global _DEFAULT_CLIENT
    with _DEFAULT_CLIENT_LOCK:
        if _DEFAULT_CLIENT is None:
            _DEFAULT_CLIENT = OllamaClient()
        return _DEFAULT_CLIENT


__all__ = [
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_RETRIES",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "OllamaCall",
    "OllamaClient",
    "OllamaConfigurationError",
    "OllamaError",
    "OllamaRequestError",
    "OllamaResponseError",
    "OllamaTimeoutError",
    "get_default_ollama_client",
]
