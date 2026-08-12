# Phase Gate Checklist

Each phase has explicit exit criteria; do not begin a phase until the prior gate passes.
Ordering is by dependency (the thesis research timeline, report.md Section 3.5, is
supplied later and will be mapped onto these phases).

## Phase 0 - Engineering Setup  [COMPLETE]
- [x] Repository skeleton created
- [x] Config system with `extends:` inheritance (`src/common/config_loader.py`)
- [x] Run-tracking / reproducibility utility (`src/common/run_context.py`)
- [x] All shipped config files load and validate
- [x] Import-smoke-test passes across every `src/` stub module
- [x] Unit test suite green (`pytest tests/unit`)
- [x] CI workflow (lint + unit tests)
- **Exit criteria:** `pytest tests/unit` passes; `config/arena_config.yaml` loads and validates. MET.

## Phase 1 - System Architecture Overview
- [ ] `PyBulletSumoEnv`, arena geometry, chassis physics
- [ ] Baseline 1 (rule-based) implemented and frozen
- [ ] Baseline 2 (PPO via SB3) trained to a stable policy and frozen
- [ ] MRA (Baseline 3) single-call reasoning path wired
- **Exit criteria:** Baseline 1 and Baseline 2 each complete a full match in-sim; PPO converges to a non-degenerate policy.

## Phase 2 - Multi-Agent Orchestration Framework
- [ ] PA (LSSD: Savitzky-Golay + IR gradient + motor deadband)
- [ ] OAA, SA, TEA agents
- [ ] HCMA (token-budget governor + emergency SA-bypass; see plan Section 6.5)
- [ ] Actuator Bridge
- [ ] LangGraph StateGraph wiring with unified state
- **Exit criteria:** a full PA -> OAA -> SA -> TEA -> Actuator Bridge cycle runs end-to-end and yields a schema-valid TacticalCommand.

## Phase 3 - SLM Inference Infrastructure
- [ ] llama.cpp deployment serving path
- [ ] SGLang rollout-sampling path
- [ ] Outlines constrained decoding wrapper
- [ ] HEL Jetson Nano profile + logger
- **Exit criteria:** HEL logs requested vs. actual time, memory, CPU quota, deadline-compliance flag; Outlines-constrained TEA output validates on first generation.

## Phase 4 - Simulation-Based Strategy Optimization
- [ ] MCTS (Selection/Expansion/Simulation/Backprop)
- [ ] LLM-as-a-Judge filtering
- [ ] DSPy recompilation + scheduler (K-batch / reward-drop triggers)
- [ ] Opponent pool + self-checkpoint manager
- [ ] Ablation strategies (No-SA / No-MCTS / No-DSPy / No-Judge)
- [ ] LoRA -> merge(fp16) -> GPTQ INT4 -> GGUF pipeline
- **Exit criteria:** one full 5,000-episode Benchmark 2 run completes, exports GGUF, and the checkpoint loads in llama.cpp.

## Phase 5a - Comparative Simulation Evaluation
- [ ] `match_runner.py` executes Blocks A-D (3,500 matches)
- [ ] Win rate (Mann-Whitney U + pp-delta), ELO (Mann-Whitney U), latency (bootstrap + Cohen's d)
- **Exit criteria:** all four blocks complete against the KPI table; results in `results/phase5a/`.

## Phase 5b - Component Ablation Study
- [ ] Same harness vs. the four ablation checkpoints (2,000 matches)
- **Exit criteria:** Cohen's d (>= 0.3 bar) computed per ablation scenario.

## Phase 5c - Real-World Validation  [BLOCKED: empirical dataset not yet received]
- [ ] `empirical_sensor_loader.py` (ToF/IR/Encoder only)
- [ ] Sensor Robustness Index, Decision Stability Index, schema-validity-under-noise, latency-compliance-under-noise
- **Exit criteria:** all confirmed-N episodes processed; four Phase 5c KPIs computed (two fixed thresholds, two descriptive).

## Phase 6 - Reporting
- [ ] `generate_report_tables.py` regenerates thesis KPI tables from `results/`
