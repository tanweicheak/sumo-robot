"""Unit tests for the deterministic Perception Agent + LSSD (Phase 2)."""

from __future__ import annotations

import unittest

import numpy as np

from src.agents.perception_agent import PerceptionAgent
from src.agents.schemas import DistanceLabel, EdgeLabel, PerceptionState


class TestPerceptionAgent(unittest.TestCase):
    def setUp(self):
        self.pa = PerceptionAgent(detect_range_m=1.4)
        self.pa.reset()
        self.n = 7

    def _step(self, tof, ir, enc=(0, 0, 0, 0)):
        return self.pa.perceive(np.array(tof, np.float32), np.array(ir, np.float32),
                                np.array(enc, np.float32))

    def test_returns_perception_state(self):
        out = self._step([1.5] * self.n, [0.29, 0.29])
        self.assertIsInstance(out, PerceptionState)
        self.assertIn("opp=", out.lssd_text)
        self.assertIn("edge=", out.lssd_text)

    def test_no_opponent_when_far(self):
        # All rays at max range -> no detection.
        out = self._step([1.5] * self.n, [0.29, 0.29])
        self.assertEqual(out.opp_distance, DistanceLabel.NONE)

    def test_near_opponent_center(self):
        tof = [1.5] * self.n
        tof[3] = 0.15   # 15 cm dead center -> near, FC
        out = self._step(tof, [0.29, 0.29])
        self.assertEqual(out.opp_distance, DistanceLabel.NEAR)
        self.assertIn("dir=FC", out.lssd_text)

    def test_edge_safe_when_ir_stable(self):
        # Constant IR -> zero approach rate -> safe.
        for _ in range(6):
            out = self._step([1.5] * self.n, [0.29, 0.29])
        self.assertEqual(out.edge, EdgeLabel.SAFE)

    def test_lssd_text_is_compact(self):
        out = self._step([1.5] * self.n, [0.29, 0.29])
        # Compact key-value form, no spaces, semicolon-separated.
        self.assertNotIn(" ", out.lssd_text)
        self.assertEqual(out.lssd_text.count(";"), 2)


if __name__ == "__main__":
    unittest.main()