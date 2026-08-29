"""Configuration for the Fast/Slow longitudinal-memory baseline."""

from __future__ import annotations

from dataclasses import dataclass


MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class MemoryConfig:
    """The complete set of tunable memory-policy parameters.

    ``memory_enabled`` controls only whether completed Slow Memory is read at
    ranking time. Both modes still update Fast Memory and commit episodes.
    """

    memory_enabled: bool = True
    tau: float = 6.0
    lambda_memory: float = 0.02
    candidate_depth: int = 1000
    embedding_model_id: str = MINILM_MODEL_ID

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.lambda_memory < 0:
            raise ValueError("lambda_memory cannot be negative")
        if self.candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")
        if not str(self.embedding_model_id).strip():
            raise ValueError("embedding_model_id must be non-empty")


def memory_config_for_mode(mode: str) -> MemoryConfig:
    """Return the M0 (write-only) or M1 (read/write) baseline config."""

    key = str(mode).upper()
    if key == "M0":
        return MemoryConfig(memory_enabled=False)
    if key == "M1":
        return MemoryConfig(memory_enabled=True)
    raise ValueError(f"Unknown memory mode {mode!r}; expected 'M0' or 'M1'")


__all__ = ["MINILM_MODEL_ID", "MemoryConfig", "memory_config_for_mode"]
