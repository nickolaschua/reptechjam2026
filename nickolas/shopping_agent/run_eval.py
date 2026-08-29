from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
experiment_dir = project_root / "experiment_1"
shared_repo = project_root / "techjam-conversational-search"

for path in (shared_repo, project_root, experiment_dir, current_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from agent import Agent


shared_evaluator = importlib.import_module("experiment_1.run_eval_v2")
shared_evaluator.Agent = Agent


def main() -> None:
    os.chdir(current_dir)
    shared_evaluator.main()


if __name__ == "__main__":
    main()
