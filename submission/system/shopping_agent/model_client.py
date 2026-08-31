"""Provider-neutral chat model contracts used by every active LLM role."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


class ModelError(RuntimeError):
    """Base class for an exhausted provider call."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        latency_seconds: float,
        attempts: int,
        role: str,
        cause_type: str,
        provider: str,
    ) -> None:
        super().__init__(message)
        self.model = str(model)
        self.latency_seconds = float(latency_seconds)
        self.attempts = int(attempts)
        self.retry_count = max(0, self.attempts - 1)
        self.role = str(role)
        self.cause_type = str(cause_type)
        self.provider = str(provider)

    def instrumentation(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "role": self.role,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "attempts": self.attempts,
            "retry_count": self.retry_count,
            "success": False,
            "error_type": type(self).__name__,
            "cause_type": self.cause_type,
        }


@dataclass(frozen=True)
class ModelCall:
    content: str
    model: str
    latency_seconds: float
    attempts: int
    role: str
    provider: str
    usage: Mapping[str, int] | None = None

    @property
    def retry_count(self) -> int:
        return max(0, self.attempts - 1)

    def instrumentation(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "role": self.role,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "attempts": self.attempts,
            "retry_count": self.retry_count,
            "success": True,
            "error_type": None,
            "cause_type": None,
            "usage": dict(self.usage or {}),
        }


@runtime_checkable
class ModelClient(Protocol):
    model: str
    provider: str

    def chat_result(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        format: str | Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        role: str = "chat",
        validator: Callable[[str], object] | None = None,
    ) -> ModelCall: ...

    def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str: ...

    def instrumentation(self) -> list[dict[str, Any]]: ...


__all__ = ["ModelCall", "ModelClient", "ModelError"]
