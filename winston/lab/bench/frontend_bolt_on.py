"""Yang Xu's live front end (experiment_1/visualizer) driving the BOLT-ON agent.

His server only ever calls Agent(catalog).reset/.respond, so this swaps the agent
symbol and serves on its own port - server.py, the HTML, and shop_agent are not
modified. Run his stock server on 8080 and this on 8081 to A/B the same messy
prompt side by side.

    python3 frontend_bolt_on.py [port]        # default 8081
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent.parent.parent
VIS = REPO / "experiment_1" / "visualizer"
for p in (BENCH, VIS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bolt_on_agent                        # noqa: E402
import server                               # noqa: E402  (Yang Xu's, untouched)

server.Agent = bolt_on_agent.Agent          # run_server reads this module global

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"[bolt-on] serving Yang Xu's UI with the bolt-on agent on port {port}")
    server.run_server(port)
