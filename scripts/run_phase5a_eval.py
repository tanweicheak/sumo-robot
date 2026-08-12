"""
scripts.run_phase5a_eval

Phase: Phase 5A
Purpose: Phase 5a: comparative simulation evaluation (Blocks A-D).

Status: STUB entry point (Phase 0). Wires config loading + run tracking so the script
    runs end-to-end today; phase-specific logic lands in Phase 5A.
"""

from __future__ import annotations

from scripts._script_common import build_run


def main() -> None:
    config, ctx = build_run(phase="phase5a", description="Phase 5a: comparative simulation evaluation (Blocks A-D).")
    print("[phase5a] config + run context ready. Phase logic not yet implemented.")
    # TODO(Phase 5A): implement the phase workflow here.


if __name__ == "__main__":
    main()
