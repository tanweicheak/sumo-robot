"""
scripts.smoke_test_slm

Phase: Phase 3
Purpose: One-shot real-inference smoke test. Loads the configured GGUF model via
    llama.cpp and runs a single TEA-style call, confirming a grammar-constrained,
    schema-valid TacticalCommand comes back. Not part of the fast dev loop - run once
    to verify the real backend, then develop against the mock.

Usage:
    python -m scripts.smoke_test_slm            # uses config/inference.yaml
    python -m scripts.smoke_test_slm --model models/phi-4-mini-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import time

from src.agents.schemas import MacroStrategy, MacroStrategyDecision, PerceptionState, TacticalCommand
from src.agents.schemas import DistanceLabel, DirectionLabel, EdgeLabel, MomentumLabel
from src.agents.tactical_execution_agent import TacticalExecutionAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the real llama.cpp SLM backend.")
    parser.add_argument("--model", default=None, help="Override GGUF model path.")
    args = parser.parse_args()

    from src.inference.llama_cpp_server import LlamaCppSLMClient
    from src.inference.factory import build_slm_client

    if args.model:
        client = LlamaCppSLMClient(model_path=args.model)
    else:
        client = build_slm_client()   # honors config/inference.yaml (set backend: llama_cpp)

    perception = PerceptionState(
        lssd_text="opp=near,dir=FC;edge=safe;mom=fwd",
        opp_distance=DistanceLabel.NEAR, opp_direction=DirectionLabel.FRONT_CENTER,
        edge=EdgeLabel.SAFE, momentum=MomentumLabel.FORWARD, opp_distance_m=0.15,
    )
    macro = MacroStrategyDecision(strategy=MacroStrategy.CHARGE)

    tea = TacticalExecutionAgent(client)
    t0 = time.perf_counter()
    cmd = tea.execute(perception, macro)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert isinstance(cmd, TacticalCommand), "TEA did not return a TacticalCommand"
    print(f"[smoke] real TEA call OK -> keyword={cmd.keyword.value}  ({dt_ms:.1f} ms)")
    print("[smoke] grammar-constrained output is schema-valid. Backend works.")


if __name__ == "__main__":
    main()