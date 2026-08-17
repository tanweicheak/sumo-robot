# Sumo-SBSO — Project Implementation Plan (Phase 0-4)

Purpose: what each phase is FOR, which files implement it, and the reasoning behind
each major design choice — not just a file list, but why the architecture looks the
way it does. Audit status noted per phase.

---

## Phase 0 — Shared Infrastructure

**Goal:** every later phase needs the same three things — load a YAML config with
inheritance, track what config/git-commit produced a given run, and give every script
a consistent CLI shape. Building this once, first, means Phase 1-5 scripts never
reinvent config loading or lose track of what produced a result.

**Files:**
- `src/common/config_loader.py` — `load_config()`, `extends:`-based YAML inheritance
- `src/common/run_context.py` — `RunContext`, run identity/provenance tracking
- `scripts/_script_common.py` — `build_run()`, the shared CLI entry point

**Design logic:**
- **Config inheritance (`extends:`) over duplication.** Every Phase 4 training variant
  (Benchmark 2, 4 ablations) shares 90% of its settings — episode counts, MCTS budgets,
  checkpoint intervals. Rather than copy-pasting a full config per variant (and risking
  them silently drifting apart), each variant config only states what's *different*
  from `_shared_defaults.yaml`.
- **`RunContext` exists because Phase 4 runs are expensive and long.** A `run_id`
  encoding variant + phase + timestamp + config-hash means you can always answer "what
  exact settings produced this checkpoint" days later, without relying on memory.
- **`build_run()`'s `extra_args` callback** (added this session) exists because
  different scripts need genuinely different CLI surfaces — a training script needs
  `--sim-budget`, an export script needs `--adapter-path` — but ALL of them still want
  `RunContext` tracking. One shared entry point, extensible per-caller, beats either
  duplicating the argparse boilerplate everywhere or forcing every script into an
  identical CLI shape that doesn't fit its actual job.

**Audit status:** Complete. 2 bugs fixed (`run_id` timestamp, non-composable `build_run()`).

---

## Phase 1 — Physics, Sensing, Baselines

**Goal:** a physics simulation the SLM pipeline can act inside, real-feeling sensor
noise/constraints (only 3 channels, no privileged global position), and two baselines
(rule-based, PPO) that don't depend on any SLM at all — so there's a working
environment and a lower bound to compare against before any language model is involved.

**Files:**
- `src/simulation/{sumo_env, arena, robot, sensors}.py` — the environment itself
- `src/preprocessing/{ir_gradient, motor_deadband, savitzky_golay_filter}.py` — signal
  processing shared by the real agent pipeline (Phase 2) and training (Phase 4)
- `src/baselines/{rule_based_controller, ppo_controller}.py` + `scripts/run_phase1_baselines.py`
  + `scripts/watch_baseline.py`

**Design logic:**
- **Exactly 3 sensor channels (ToF, IR, encoder), no IMU, no privileged position.** This
  is a deliberate constraint (`project_config.yaml`'s `sensor_channels`/
  `scope_exclusions`), not an oversight — the whole point of the research is whether an
  SLM can reason well from the SAME limited, physically-realistic sensor picture a real
  robot would have. Giving the agent global ring-position knowledge would make the
  comparison against a real deployment meaningless.
- **IR mounted below the chassis hull, filtered by `self_body_id`.** A raycast sensor
  mounted naively inside a robot's own collision geometry self-detects — this was a
  real bug caught early in the project (see memory: "IR probe self-collision"). The fix
  (mount below hull + filter self-hits) is now the standard pattern `sensors.py` uses.
- **Rule-based Baseline 1 is deliberately kept simple**, even after this session's
  enhancements (arc-search coverage, ray-drift lead prediction) — it's meant to stay a
  fair, legible "conventional lower bound," not accumulate adaptive sophistication that
  would blur the contrast SBSO's results need to show.
- **PPO (Baseline 2) needs opponent diversity to be a meaningful baseline at all** — a
  policy trained against one frozen opponent learns to exploit that opponent
  specifically, not general pushing competence. `RandomizedOpponentWrapper` (added this
  session) exists for exactly this reason.

**Audit status:** Complete. Multiple bugs fixed (see `PHASE_0_1_2_FILE_SUMMARY.md` for the
full list) — most significant: `SensorSpec`/`RobotSpec` chassis-dimension desync, missing
`scipy` dependency, PPO's opponent-diversity + reset bugs.

---

## Phase 2 — Multi-Agent Reasoning Layer

**Goal:** the actual multi-agent architecture the research is centered on — decompose
"decide what to do" into Perception → Opponent Analysis → Strategy → Tactical
Execution, instead of one monolithic call, and give each stage a narrow, well-typed
job.

**Files:**
- `src/agents/schemas.py` — the shared typed vocabulary every agent speaks
- `src/agents/perception_agent.py` (PA) — raw sensors → semantic LSSD text
- `src/data/lssd_encoder.py` — the actual sensor→text mapping PA uses
- `src/agents/opponent_analysis_agent.py` (OAA) — LSSD history → opponent behavior classification
- `src/agents/strategy_agent.py` (SA) — LSSD + OAA → macro-strategy
- `src/agents/tactical_execution_agent.py` (TEA) — macro-strategy → tactical keyword
- `src/agents/actuator_bridge.py` — tactical keyword → PWM (deterministic, not SLM)
- `src/agents/hardware_constraint_monitor.py` (HCMA) — token-budget/latency governor
- `src/agents/monolithic_reasoning_agent.py` (MRA) — the one-shot alternative (Baseline 3)
- `src/orchestration/{graph_builder, state}.py` — wires PA→OAA→SA→TEA→bridge into a
  real LangGraph, and `SumoController`, the live-match driver

**Design logic:**
- **Why decompose into 4 stages instead of 1 call?** This is the central research
  question (multi-agent vs. monolithic), so Phase 2 has to implement BOTH sides fairly
  — the multi-agent pipeline (PA/OAA/SA/TEA) AND the one-shot alternative (MRA) — using
  the SAME final action vocabulary (`TacticalKeyword`) and the SAME `ActuatorBridge`, so
  the comparison is about reasoning architecture, not action-space differences.
- **Pipelined staleness (SA reads the PREVIOUS frame's OAA output, not the current
  one).** Running OAA and SA sequentially every frame would double per-frame SLM
  latency. Running them in parallel needs SA to use *something* for opponent behavior
  — using last frame's classification (nearly always still valid a frame later, given
  a 50ms cycle) trades a small staleness cost for roughly half the latency.
- **TEA emits a small categorical keyword, not raw PWM.** Keeps the SLM's job narrow
  (pick 1 of 7 keywords) and moves the actual physical mapping into deterministic code
  (`ActuatorBridge`) — the SLM never has to reason about motor calibration, only tactics.
- **HCMA is fully built but deliberately NOT wired into the live cycle yet.** Its job is
  throttling based on REAL hardware latency; there's no genuine signal to throttle
  against until Phase 5c's real sensor/timing data exists. Wiring it earlier would just
  be reacting to a mocked number.

**Audit status:** Complete. 1 real bug fixed (OAA prompt history-order contradiction);
MRA implemented this session (was a stub).

---

## Phase 3 — SLM Inference Backends

**Goal:** one abstraction (`SLMClient`) that every agent codes against, with three
interchangeable implementations — mock (fast dev/CI), llama.cpp (local/edge, GGUF),
SGLang (cloud, HF checkpoints, concurrent batching) — so agent code never changes when
you swap where the SLM actually runs.

**Files:**
- `src/inference/slm_client.py` — the `SLMClient` contract + `MockSLMClient`/`ScriptedSLMClient`
- `src/inference/grammar.py` — GBNF/regex constrained-decoding grammar construction
- `src/inference/llama_cpp_server.py` — real in-process llama.cpp backend
- `src/inference/sglang_server.py` — real HTTP SGLang backend
- `src/inference/factory.py` — config-driven backend selection
- `src/inference/outlines_decoder.py` — **effectively dead code**, see below

**Design logic:**
- **Every agent's SLM output is ONE categorical token, not a structured object.**
  Given each agent's job is narrow (OAA picks 1 of N behaviors, SA picks 1 of 5
  strategies, TEA picks 1 of 7 keywords), the constrained-decoding problem is "force
  exactly one of a short list of tokens" — which native grammar support (GBNF in
  llama.cpp, regex in SGLang) solves with near-zero overhead, directly in the inference
  engine. This is why `outlines_decoder.py` was scaffolded but never finished — Outlines
  solves a broader problem (whole-schema validation) this project doesn't actually have,
  at a real latency cost this project's 50ms budget can't easily absorb. **Recommend
  retiring it rather than implementing it** — the job it was meant to do is already done,
  better-suited to the actual constraint (narrow, latency-critical), by the grammar
  files that already work.
- **SGLang exists specifically for MCTS's concurrent rollout scoring**, not as "the
  cloud version of llama.cpp" in general. `generate_structured_batch()`'s whole reason
  to exist is firing many requests concurrently so SGLang's continuous batcher can
  combine them server-side — the single-call path gets none of that benefit and is only
  for one-off calls.

**Audit status:** Complete. 2 bugs fixed this session (`grammar.py`'s stray docstring
artifact, `sglang_server.py`'s `call_count` race condition). `outlines_decoder.py`'s fate
still needs your final word (retire vs. leave as a stub).

---

## Phase 4 — SBSO (the core research contribution)

**Goal:** teach the SLM good sumo tactics WITHOUT a labeled dataset, by using physics
simulation + MCTS + an LLM Judge to generate a training signal, then DSPy to turn that
signal into better prompts and LoRA to bake it into the model — evaluated against 4
ablations (each removing one component) to isolate what actually matters.

**Files — training core:**
- `src/sbso/macro_executor.py` — `MacroStrategyExecutor` (fast, deterministic strategy→PWM,
  used for BOTH sides during hypothetical MCTS search — see D2 below)
- `src/sbso/mcts.py` — the tree search itself
- `src/sbso/judge.py` — `MockJudge`/`LlamaCppJudge`/`SGLangJudge`, position/branch scoring
- `src/sbso/simulation_backend.py` — `PyBulletSimulationBackend`, the real-physics MCTS backend
- `src/sbso/opponent_pool.py` — training opponent sampling (baseline1/baseline2/self_checkpoint)
- `src/sbso/self_checkpoint_manager.py` — periodic snapshots of the model's own past prompts
- `src/sbso/recompilation_scheduler.py` — when to trigger a DSPy recompile
- `src/sbso/dspy_compiler.py` — `MockDSPyCompiler`/`RealDSPyCompiler`, the actual prompt optimization
- `src/sbso/match_trainer.py` — `MatchLevelSBSOTrainer`, the canonical training loop (full matches)
- `src/sbso/ablation_strategies.py` — **not yet audited this session**
- `scripts/run_phase4_pilot.py` — the one production/cloud entry point (pilot AND full scale)
- `scripts/run_phase4_stage3_local.py` — local Mac equivalent (SGLang doesn't run on macOS)
- `scripts/run_phase4_training.py` — Stage 1, mock-everything plumbing validation

**Files — export/deployment:**
- `src/finetuning/merge_adapters.py` — **not yet audited this session**
- `src/finetuning/export_gguf.py` — merged model → GGUF f16 (edge branch)
- `src/finetuning/quantize_gguf.py` — GGUF f16 → Q4_K_M (edge branch, cont'd)
- `src/finetuning/quantize_gptq.py` — merged model → GPTQ INT4 (cloud branch, independent)
- `src/finetuning/calibration_texts.py` — real LSSD text extraction for GPTQ calibration
- `src/finetuning/lora_finetune.py` — **not yet audited this session**
- `src/finetuning/cost_projection.py` — **not yet audited this session**
- `scripts/run_export_pipeline.py` — orchestrates the whole merge→export sequence
- `scripts/launch_sglang_servers.sh` — starts both SGLang servers (agent + judge)

**Design logic — the decisions that shape everything above:**
- **"Episode" means one full match, not one decision.** This single scope correction is
  why `MatchLevelSBSOTrainer` plays real matches to a natural conclusion (win/loss/draw)
  rather than treating one MCTS decision as a whole training unit — a match genuinely
  contributes many (state, strategy) pairs, not one.
- **MCTS rollouts NEVER use live SLM inference, for either side (D2).** Real inference
  during tree search would mean dozens of rollouts × real latency per single decision —
  far too slow. `MacroStrategyExecutor` gives MCTS a fast, deterministic stand-in for
  BOTH the agent's and the opponent's hypothetical moves; only the real, committed match
  continuation uses whatever the opponent actually is (baseline, or eventually a live
  self-checkpoint).
- **PyBullet state snapshots need explicit lifecycle management.** `saveState()`
  doesn't garbage-collect, and it only captures physics — not Python-side counters like
  `_step_count`. Both were real, confirmed bugs this session; both are now handled by
  `PyBulletSimulationBackend`'s own tracked-id bookkeeping, transparently, for whichever
  trainer uses it.
- **Self-checkpoint opponents exist so training doesn't stagnate against fixed
  opponents forever** — periodically freezing the model's own current prompt program as
  a future opponent is the same idea as PPO's opponent-diversity fix in Phase 1, applied
  to the SBSO side.
- **Target-proportion opponent sampling, not pure uniform random.** Pure random sampling
  systematically under-samples self-checkpoint opponents (they're excluded during
  warmup with no compensation afterward) — deficit-weighted sampling actually converges
  on the config's intended 1667/1667/1666 split.
- **DSPy's few-shot selection must be match-aware.** Now that one episode is a full
  match (potentially 50-100+ decisions), a flat "most recent N pairs" slice could be
  entirely one match/one opponent — spreading selection across several recent matches
  is what actually gives DSPy cross-opponent diversity to compile against.
- **Quantization is two independent, parallel branches, not a chain.** GPTQ (cloud/SGLang)
  and GGUF/Q4_K_M (edge/llama.cpp) are incompatible formats — both start from the same
  merged fp16 model, but neither's output feeds the other. This was a real, confirmed
  bug (D4) before this session's fix.
- **Crash-safety (not crash-resumption) for long cloud runs.** `progress.json` +
  `training_pairs.jsonl` are written incrementally so an interrupted RunPod pod loses at
  most one checkpoint-interval's worth of progress — but the run itself doesn't
  currently pick back up automatically from a saved point; that's a further, unbuilt
  piece of work if you want it.

**Audit status: partial.** Extensively fixed through this session's natural work (the
`match_trainer.py`/`run_phase4_pilot.py` consolidation was the single largest change of
the whole project), but `ablation_strategies.py`, `cost_projection.py`,
`lora_finetune.py`, `merge_adapters.py` have never actually been read yet — genuinely
open before Phase 4 can be called fully audited.

---

## Not yet started: Phase 5 (Evaluation)

No files read, no scripts built. Will need: the 5-condition comparison harness,
`src/evaluation/*` (elo, win_rate, latency_profiler, decision_stability_index,
sensor_robustness_index, schema_validity_checker — all currently just names from
`graph.json`, unread), `src/statistics/*` (bootstrap, cohens_d, mann_whitney,
similarly unread), and the actual Phase 5a/5b/5c configs.
