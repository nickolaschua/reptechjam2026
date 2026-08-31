"""Run every registered experiment and refresh results/*.json."""
from __future__ import annotations

import importlib
import time

from common import write_result

EXPERIMENTS = [
    ("exp01_catalog_profile", "what is in the catalog and what can be filtered on"),
    ("exp02_simulator_leakage", "how much the public simulator gives away"),
    ("exp03_scoring_ablation", "which scoring components earn their place"),
    ("exp04_robustness", "how far the system falls under paraphrase"),
    ("exp05_agent_diagnostics", "why experiment_1 fails on messy input"),
]

if __name__ == "__main__":
    manifest = []
    for name, question in EXPERIMENTS:
        print(f"\n=== {name}: {question}")
        started = time.time()
        module = importlib.import_module(name)
        write_result(name, module.main())
        manifest.append({"experiment": name, "question": question,
                         "seconds": round(time.time() - started, 1),
                         "result": f"results/{name}.json"})
    write_result("_manifest", {"experiments": manifest})
    print("\nall experiments complete")
