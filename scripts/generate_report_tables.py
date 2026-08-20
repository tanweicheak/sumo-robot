"""
scripts.generate_report_tables

Phase: Reporting
Purpose: Phase 6: regenerate thesis KPI tables from results/.

Status: STUB entry point (Phase 0). Wires config loading + run tracking so the script
    runs end-to-end today; phase-specific logic lands in Reporting.
"""

from __future__ import annotations

from scripts._script_common import build_run


def main() -> None:
    config, ctx, _args = build_run(phase="reporting", description="Phase 6: regenerate thesis KPI tables from results/.")
    print("[reporting] config + run context ready. Phase logic not yet implemented.")
    # TODO(Reporting): implement the phase workflow here.


if __name__ == "__main__":
    main()