# Legacy QLMP archive

Archived on 2026-08-30 when the live Nickolas shopping agent moved to gated vector memory.

This tree contains the disconnected QLMP library, adapter and integration boundary, projector/portability/masked/relevance experiments, their fixtures and results, and associated tests. Archived Python tests use a `.py.txt` suffix and `pytest.ini` excludes this directory, so none of these files participate in active imports or test discovery.

The archive is historical evidence only. Active code must not import it. The longitudinal runner, `users_40.json`, fixture validation utilities, embedding backends, and generic short-term tests remain under `nickolas/shopping_agent/`.

Directory map:

- `library/`: former `nickolas/memory/qlmp` package and unit tests.
- `integration/`: former live QLMP adapter and integration module.
- `experiments/`: projector, portability, masked-memory, relevance, filtered-universe, and obsolete shadow-parity programs.
- `fixtures/`: experiment-only fixtures.
- `results/`: frozen experiment outputs.
- `documentation/`: legacy experiment documentation.
- `tests/`: disconnected integration/experiment tests, renamed to prevent discovery.
