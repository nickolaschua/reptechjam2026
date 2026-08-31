"""Compatibility entrypoint for the canonical M0_OPENAI runner."""

from __future__ import annotations

try:
    from .run_m0 import main
except ImportError:
    from run_m0 import main


if __name__ == "__main__":
    main()
