"""
tests.unit.test_config_loader

Phase: Phase 0
Purpose: Verify config loading, the `extends:` deep-merge, and required-key
    validation. Runs under both `python -m unittest` and `pytest`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.common.config_loader import ConfigError, load_config, require_keys


class TestConfigLoader(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_plain_config(self) -> None:
        path = self._write("plain.yaml", "a: 1\nb:\n  c: 2\n")
        cfg = load_config(path)
        self.assertEqual(cfg, {"a": 1, "b": {"c": 2}})

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.tmp / "does_not_exist.yaml")

    def test_extends_deep_merge(self) -> None:
        self._write(
            "_base.yaml",
            "episodes_total: 5000\n"
            "opponent_pool:\n"
            "  warmup_episodes: 500\n"
            "  full_run_targets:\n"
            "    baseline1: 1667\n"
            "    self_checkpoint: 1666\n",
        )
        child = self._write(
            "child.yaml",
            "extends: _base.yaml\n"
            "variant_name: benchmark2_full_sbso\n"
            "opponent_pool:\n"
            "  full_run_targets:\n"
            "    self_checkpoint: 2000\n",  # override one nested leaf only
        )
        cfg = load_config(child)
        self.assertEqual(cfg["episodes_total"], 5000)          # inherited
        self.assertEqual(cfg["variant_name"], "benchmark2_full_sbso")  # child-only
        self.assertEqual(cfg["opponent_pool"]["warmup_episodes"], 500)  # inherited nested
        self.assertEqual(cfg["opponent_pool"]["full_run_targets"]["baseline1"], 1667)  # inherited leaf
        self.assertEqual(cfg["opponent_pool"]["full_run_targets"]["self_checkpoint"], 2000)  # overridden leaf
        self.assertNotIn("extends", cfg)  # extends stripped

    def test_extends_child_overrides_scalar(self) -> None:
        self._write("_base.yaml", "match_count_per_pairing: 500\n")
        child = self._write(
            "child.yaml", "extends: _base.yaml\nmatch_count_per_pairing: 800\n"
        )
        cfg = load_config(child)
        self.assertEqual(cfg["match_count_per_pairing"], 800)

    def test_require_keys_dotted_path(self) -> None:
        cfg = {"arena": {"radius_m": 1.5}}
        require_keys(cfg, ["arena.radius_m"])  # should not raise
        with self.assertRaises(ConfigError):
            require_keys(cfg, ["arena.diameter_m"], context="arena_config")


if __name__ == "__main__":
    unittest.main()
