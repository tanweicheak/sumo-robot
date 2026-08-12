"""
scripts.run_phase4_ablation

Phase: Phase 4
Purpose: Phase 4: run a single ablation variant (No-SA / No-MCTS / No-DSPy / No-Judge).

Status: STUB entry point (Phase 0). Wires config loading + run tracking so the script
    runs end-to-end today; phase-specific logic lands in Phase 4.
"""

from __future__ import annotations

from scripts._script_common import build_run


def main() -> None:
    config, ctx = build_run(phase="phase4", description="Phase 4: run a single ablation variant (No-SA / No-MCTS / No-DSPy / No-Judge).")
    print("[phase4] config + run context ready. Phase logic not yet implemented.")
    # TODO(Phase 4): implement the phase workflow here.


if __name__ == "__main__":
    main()
