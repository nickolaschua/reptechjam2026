from __future__ import annotations

import json
import math
import unittest

import numpy as np

from nickolas.memory.qlmp import (
    BaselineConfig,
    MemoryItem,
    MemoryPolarity,
    MemorySource,
    SteeringConfig,
    bound_query_shift,
    build_cosine_memory_baseline,
)


class DiagnosticSerializationTests(unittest.TestCase):
    def test_baseline_diagnostics_expose_selection_and_serialize(self) -> None:
        q = np.asarray([1.0, 0.0, 0.0])
        memories = [
            MemoryItem(
                id="selected",
                text="selected",
                embedding=[0.8, 0.6, 0.0],
                source=MemorySource.PURCHASE_EPISODE,
                polarity=MemoryPolarity.POSITIVE,
                scope="footwear",
                confidence=0.7,
            ),
            MemoryItem(
                id="filtered",
                text="filtered",
                embedding=[0.2, 0.0, np.sqrt(0.96)],
                source=MemorySource.CLICK,
                polarity=MemoryPolarity.NEUTRAL,
                scope="footwear",
            ),
        ]
        result = build_cosine_memory_baseline(
            q,
            memories,
            query_scope="footwear",
            config=BaselineConfig(memory_top_k=1, cosine_threshold=0.1),
        )
        payload = result.to_dict()
        self.assertEqual(payload["mode"], "cosine")
        self.assertEqual(payload["selected_memory_ids"], ["selected"])
        self.assertEqual(payload["memory_diagnostics"][0]["selection_rank"], 1)
        self.assertEqual(payload["memory_diagnostics"][0]["aggregation_weight"], 1.0)
        self.assertFalse(payload["memory_diagnostics"][1]["selected"])
        json.dumps(payload, allow_nan=False)
        self.assertFalse(result.aggregate_delta.flags.writeable)

    def test_steering_diagnostics_expose_clipping_and_serialize(self) -> None:
        result = bound_query_shift(
            [1.0, 0.0],
            [0.0, 1.0],
            config=SteeringConfig(
                beta=math.tan(math.radians(20.0)), max_shift_deg=10.0
            ),
        )
        payload = result.to_dict()
        diagnostics = payload["diagnostics"]
        self.assertTrue(diagnostics["clipped"])
        self.assertAlmostEqual(diagnostics["unclipped_angle_deg"], 20.0)
        self.assertAlmostEqual(diagnostics["actual_shift_deg"], 10.0)
        self.assertLess(diagnostics["applied_beta"], diagnostics["requested_beta"])
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
