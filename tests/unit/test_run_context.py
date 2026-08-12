"""
tests.unit.test_run_context

Phase: Phase 0
Purpose: Verify run-id construction, config hashing (deterministic + change-sensitive),
    and metadata persistence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.common.run_context import RunContext


class TestRunContext(unittest.TestCase):
    def _make(self, config: dict | None = None, variant: str = "benchmark2_full_sbso") -> RunContext:
        return RunContext(
            variant_name=variant,
            phase="phase4",
            resolved_config=config if config is not None else {"episodes_total": 5000},
        )

    def test_run_id_populated(self) -> None:
        ctx = self._make()
        self.assertTrue(ctx.run_id)
        self.assertIn("benchmark2_full_sbso", ctx.run_id)
        self.assertIn("phase4", ctx.run_id)
        self.assertTrue(ctx.config_hash)
        self.assertIn(ctx.config_hash, ctx.run_id)

    def test_config_hash_deterministic(self) -> None:
        a = self._make({"a": 1, "b": 2})
        b = self._make({"b": 2, "a": 1})  # key order should not matter
        self.assertEqual(a.config_hash, b.config_hash)

    def test_config_hash_changes_with_config(self) -> None:
        a = self._make({"episodes_total": 5000})
        b = self._make({"episodes_total": 6000})
        self.assertNotEqual(a.config_hash, b.config_hash)

    def test_save_writes_json(self) -> None:
        ctx = self._make()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = ctx.save(tmp)
            self.assertTrue(out_path.exists())
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
            self.assertEqual(data["variant_name"], "benchmark2_full_sbso")
            self.assertEqual(data["resolved_config"]["episodes_total"], 5000)
            self.assertIn("git_commit", data)
            self.assertIn("created_at", data)


if __name__ == "__main__":
    unittest.main()
