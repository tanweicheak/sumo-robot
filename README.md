# Sumo-SBSO Framework

Multi-agent Small Language Model (SLM) framework with Simulation-Based Strategy
Optimization (SBSO) for autonomous sumo robotics. Research codebase for the WQD 7025
Data Science Research Project.

The framework decomposes tactical decision-making across specialized Phi-4-mini agents
(Opponent Analysis, Strategy, Tactical Execution) orchestrated with LangGraph, trains
them through an offline MCTS + LLM-as-a-Judge + DSPy loop with LoRA fine-tuning, and
evaluates the result against rule-based and PPO baselines under a Jetson Nano Hardware
Emulation Layer.

## Status

Phase 0 (engineering setup) is complete: repository scaffold, config system, run
tracking, tests, and CI are in place. Phase 1+ module bodies are stubbed and land in
their target phases. See `docs/phase_gate_checklist.md`.

## Requirements

- Python 3.10+
- Phase 0 needs only `pyyaml`, `pytest`, `ruff`. Heavier stacks (PyBullet,
  Stable-Baselines3, transformers, DSPy, SGLang, llama.cpp, etc.) install per phase -
  see `requirements.txt`. Resource-heavy training/simulation is intended to run on
  cloud GPU rather than locally.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -r requirements.txt
cp .env.example .env             # fill in paths/keys as phases require
```

## Running tests

```bash
pytest tests/unit -ra            # Phase 0 suite (no ML/sim deps needed)
# or, without pytest installed:
python -m unittest discover -s tests/unit -p "test_*.py"
```

## Running a phase entry point

Every phase script loads a YAML config, resolves `extends:` inheritance, and records a
traceable run id (config hash + git commit) before executing:

```bash
python -m scripts.run_phase4_training \
    --config config/training/phase4_full_sbso.yaml \
    --results-dir results/phase4
```

## Layout

```
config/     YAML configs (project, arena/LSSD, HCMA policy, HEL profile, training, eval)
src/        source modules, grouped by subsystem (simulation, agents, sbso, ...)
scripts/    phase entry points
tests/      unit + integration tests
data/       raw / simulated / processed (git-ignored, dirs kept via .gitkeep)
checkpoints/ per-variant model artifacts (git-ignored)
results/    per-phase evaluation outputs (git-ignored)
docs/       architecture, KPI definitions, phase-gate checklist
```

## Key design references

- Full implementation plan and the resolution log for all specification ambiguities:
  `sumo-sbso-implementation-plan.md` (kept alongside this repo).
- Model taxonomy: Baseline 1 (rule-based), Baseline 2 (PPO), Baseline 3 (Monolithic
  SLM / MRA), Benchmark 1 (Multi-Agent zero-shot), Benchmark 2 (Multi-Agent + SBSO),
  plus four ablation variants.
- The 50ms decision window is a pilot-calibrated design target, not a literature-proven
  constraint. See `docs/kpi_definitions.md`.
