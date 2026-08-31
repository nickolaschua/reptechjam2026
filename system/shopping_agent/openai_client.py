"""OpenAI Responses API implementation of the shared model-client contract."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from threading import Lock
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

try:
    from .model_client import ModelCall, ModelError
except ImportError:
    from model_client import ModelCall, ModelError

DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0
DEFAULT_OPENAI_RETRIES = 1
OpenAITransport = Callable[[str, bytes, float, Mapping[str, str]], object]


class OpenAIConfigurationError(ModelError): pass
class OpenAIRequestError(ModelError): pass
class OpenAITimeoutError(OpenAIRequestError): pass
class OpenAIResponseError(ModelError): pass


class OpenAIClient:
    provider = "openai"

    def __init__(self, *, api_key: str, model: str = DEFAULT_OPENAI_CHAT_MODEL,
                 timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
                 retries: int = DEFAULT_OPENAI_RETRIES,
                 transport: OpenAITransport | None = None,
                 base_url: str = "https://api.openai.com/v1") -> None:
        try:
            self.api_key = str(api_key).strip()
            self.model = str(model).strip()
            self.timeout_seconds = float(timeout_seconds)
            self.base_url = str(base_url).strip().rstrip("/")
            if not self.api_key: raise ValueError("OPENAI_API_KEY is required when TEST_MODE=true")
            if not self.model: raise ValueError("OPENAI_CHAT_MODEL must be non-empty")
            if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than zero")
            if isinstance(retries, bool) or int(retries) != retries or retries < 0:
                raise ValueError("retries must be a non-negative integer")
            self.retries = int(retries)
        except (TypeError, ValueError) as exc:
            raise OpenAIConfigurationError(str(exc), model=str(model), latency_seconds=0,
                attempts=0, role="configuration", cause_type=type(exc).__name__, provider=self.provider) from exc
        self._transport = transport or self._urlopen_transport
        self._calls: list[dict[str, Any]] = []
        self._lock = Lock()

    @staticmethod
    def _urlopen_transport(url: str, body: bytes, timeout: float,
                           headers: Mapping[str, str]) -> bytes:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    @staticmethod
    def _decode(response: object) -> tuple[str, str | None, dict[str, int]]:
        if isinstance(response, bytes): response = response.decode("utf-8")
        if isinstance(response, str): response = json.loads(response)
        if not isinstance(response, Mapping): raise ValueError("OpenAI response must be a JSON object")
        content = response.get("output_text")
        if not isinstance(content, str) or not content.strip():
            texts: list[str] = []
            for output in response.get("output", []) if isinstance(response.get("output"), list) else []:
                if not isinstance(output, Mapping): continue
                for part in output.get("content", []) if isinstance(output.get("content"), list) else []:
                    if isinstance(part, Mapping) and part.get("type") in {"output_text", "text"}:
                        text = part.get("text")
                        if isinstance(text, str): texts.append(text)
            content = "".join(texts)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI response contains no output text")
        usage_raw = response.get("usage")
        usage = {str(k): int(v) for k, v in usage_raw.items()
                 if isinstance(v, int)} if isinstance(usage_raw, Mapping) else {}
        actual_model = response.get("model")
        return content.strip(), actual_model if isinstance(actual_model, str) else None, usage

    @staticmethod
    def _text_format(value: str | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return {"type": "json_schema", "name": "shopping_agent_output",
                    "strict": True, "schema": dict(value)}
        return {"type": "json_object"}

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()

    def _record(self, trace: dict[str, Any]) -> None:
        with self._lock: self._calls.append(deepcopy(trace))

    def chat_result(self, messages: Sequence[Mapping[str, str]], *,
                    format: str | Mapping[str, Any] | None = None,
                    options: Mapping[str, Any] | None = None, role: str = "chat",
                    validator: Callable[[str], object] | None = None) -> ModelCall:
        normalized = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        if not normalized: raise ValueError("messages must be non-empty")
        payload: dict[str, Any] = {"model": self.model, "input": normalized}
        opts = dict(options or {})
        if "num_predict" in opts: payload["max_output_tokens"] = int(opts.pop("num_predict"))
        if "temperature" in opts: payload["temperature"] = opts.pop("temperature")
        payload.update(opts)
        if format is not None: payload["text"] = {"format": self._text_format(format)}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter(); attempts = self.retries + 1
        last: Exception | None = None; response_error = False
        for attempt in range(1, attempts + 1):
            try:
                raw = self._transport(f"{self.base_url}/responses", body, self.timeout_seconds, headers)
                content, model, usage = self._decode(raw)
                if validator is not None: validator(content)
                elif format is not None: json.loads(content)
                call = ModelCall(content, model or self.model, time.perf_counter()-started,
                                 attempt, role, self.provider, usage)
                self._record(call.instrumentation()); return call
            except Exception as exc:
                last = exc
                response_error = isinstance(exc, (ValueError, TypeError, KeyError, json.JSONDecodeError))
        error_cls: type[ModelError] = (OpenAIResponseError if response_error else
            OpenAITimeoutError if last is not None and self._is_timeout(last) else OpenAIRequestError)
        error = error_cls(f"OpenAI {role} call failed after {attempts} attempts: {last}",
            model=self.model, latency_seconds=time.perf_counter()-started, attempts=attempts,
            role=role, cause_type=type(last).__name__ if last else "UnknownError", provider=self.provider)
        self._record(error.instrumentation()); raise error from last

    def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        return self.chat_result(messages, **kwargs).content

    def instrumentation(self) -> list[dict[str, Any]]:
        with self._lock: return deepcopy(self._calls)


__all__ = ["OpenAIClient", "OpenAIConfigurationError", "OpenAIRequestError",
           "OpenAIResponseError", "OpenAITimeoutError"]
