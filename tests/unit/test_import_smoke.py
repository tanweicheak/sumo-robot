"""
tests.unit.test_import_smoke

Phase: Phase 0
Purpose: Every scaffolded stub under src/ must be importable with only the standard
    library + PyYAML present (no pybullet/transformers/etc. at module import time).
    This enforces the Phase 0 rule that heavy third-party imports are deferred to the
    body of functions/classes, not module top-level, so CI stays fast and green
    before those dependencies are installed.
"""

from __future__ import annotations

import importlib
import pkgutil
import unittest
from pathlib import Path

import src

SRC_ROOT = Path(src.__file__).resolve().parent


def _iter_module_names() -> list[str]:
    names: list[str] = []
    for mod in pkgutil.walk_packages([str(SRC_ROOT)], prefix="src."):
        names.append(mod.name)
    return sorted(names)


class TestImportSmoke(unittest.TestCase):
    def test_all_src_modules_import(self) -> None:
        failures: list[str] = []
        for name in _iter_module_names():
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 - we want to report all failures
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [], msg="Un-importable modules:\n" + "\n".join(failures))

    def test_found_expected_module_count(self) -> None:
        # Sanity floor: the scaffold should expose a substantial number of modules.
        self.assertGreaterEqual(len(_iter_module_names()), 40)


if __name__ == "__main__":
    unittest.main()
