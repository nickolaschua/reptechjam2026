from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from nickolas.experiments.experiment_10_xtr_warp_retrieval import (
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_RESULT_COUNT,
    _safe_archive_infos,
    _validate_ranking_row,
    promotion_gates,
)


class Experiment10Tests(unittest.TestCase):
    def ranking_row(self) -> dict:
        query = "electronics test query"
        return {
            "query_id": 0,
            "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "results": [[pid, float(EXPECTED_RESULT_COUNT - pid)] for pid in range(EXPECTED_RESULT_COUNT)],
        }

    def test_ranking_contract_accepts_complete_deterministic_top_1000(self) -> None:
        query, pids, scores = _validate_ranking_row(self.ranking_row(), 0)
        self.assertEqual(query, "electronics test query")
        self.assertEqual(len(pids), EXPECTED_RESULT_COUNT)
        self.assertEqual(len(scores), EXPECTED_RESULT_COUNT)

    def test_ranking_contract_rejects_duplicate_pid(self) -> None:
        row = self.ranking_row()
        row["results"][-1][0] = row["results"][-2][0]
        with self.assertRaisesRegex(RuntimeError, "duplicate PIDs"):
            _validate_ranking_row(row, 0)

    def test_ranking_contract_rejects_wrong_tie_break(self) -> None:
        row = self.ranking_row()
        row["results"][0] = [10, 1000.0]
        row["results"][1] = [9, 1000.0]
        with self.assertRaisesRegex(RuntimeError, "order/tie-break"):
            _validate_ranking_row(row, 0)

    def test_ranking_contract_rejects_pid_outside_catalog(self) -> None:
        row = self.ranking_row()
        row["results"][0][0] = EXPECTED_DOCUMENT_COUNT
        with self.assertRaisesRegex(RuntimeError, "PID out of range"):
            _validate_ranking_row(row, 0)

    def test_promotion_requires_strict_score_improvement(self) -> None:
        equal = promotion_gates(
            {"technical_score": 0.8},
            {"technical_score": 0.8},
            {"hard_failure_rescues": 2, "regressions": 1},
        )
        self.assertFalse(equal["technical_score_strictly_beats_experiment_07_bm25"])
        better = promotion_gates(
            {"technical_score": 0.800001},
            {"technical_score": 0.8},
            {"hard_failure_rescues": 2, "regressions": 1},
        )
        self.assertTrue(all(better.values()))

    def test_promotion_rejects_no_rescue_or_excess_regressions(self) -> None:
        no_rescue = promotion_gates(
            {"technical_score": 0.9},
            {"technical_score": 0.8},
            {"hard_failure_rescues": 0, "regressions": 0},
        )
        self.assertFalse(no_rescue["rescues_at_least_one_exact_hard_failure"])
        regression = promotion_gates(
            {"technical_score": 0.9},
            {"technical_score": 0.8},
            {"hard_failure_rescues": 1, "regressions": 2},
        )
        self.assertFalse(regression["regressions_do_not_exceed_rescues_vs_exact"])

    def test_archive_paths_reject_traversal_and_windows_separators(self) -> None:
        for unsafe in ("../escape", "safe/../../escape", "..\\escape", "C:/escape", "safe:stream"):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as temporary:
                archive_path = Path(temporary) / "unsafe.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(unsafe, b"x")
                with zipfile.ZipFile(archive_path) as archive:
                    with self.assertRaisesRegex(RuntimeError, "Unsafe Colab output path"):
                        _safe_archive_infos(archive)


if __name__ == "__main__":
    unittest.main()
