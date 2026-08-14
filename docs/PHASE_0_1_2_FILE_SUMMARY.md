# Phase 0, 1, 2 — File Summary

Role, usage, call sites, I/O, and audit remarks for every file covered so far.
All files in this document have been read in full and audited this session.

---

## Phase 0 — Shared Infrastructure

### `src/common/config_loader.py`
- **Role:** YAML config loading with `extends:`-based inheritance (child overwrites parent scalars/lists; dicts deep-merge).
- **Called by:** Every `*.py` file that loads a YAML config — `sumo_env.py`, `arena.py`, `robot.py`, `actuator_bridge.py`, `hardware_constraint_monitor.py`, `lssd_encoder.py`, `run_phase4_pilot.py`, `_script_common.py`, etc. Foundational, most-depended-on file in the project.
- **Input:** `load_config(path: str | Path) -> dict` — a YAML file path.
- **Output:** A merged `dict`, fully resolved through any `extends:` chain.
- **Remarks:** Audited, no bugs. Uses `yaml.safe_load` correctly. Circular-`extends` detection is depth-exhaustion-based (works, not elegant) — cosmetic note only.

### `src/common/run_context.py`
- **Role:** `RunContext` dataclass — tracks `run_id`, `git_commit`, `config_hash`, `created_at` for any script run via `build_run()`. `.save(results_dir)` writes `run_context.json`.
- **Called by:** `_script_common.py`'s `build_run()`.
- **Input:** `variant_name`, `phase`, `resolved_config` (constructor args).
- **Output:** `run_id` (string), `.save()` writes a JSON file.
- **Remarks:** **Fixed this session** — `run_id`'s timestamp had a dead `.replace("+0000","Z")` call (order-of-operations bug); now uses `strftime()` directly. Verified live.

### `scripts/_script_common.py`
- **Role:** `build_run(phase, description, extra_args=None) -> (config, ctx, args)` — the shared CLI/config/tracking entry point for every "real" script.
- **Called by:** `run_phase4_training.py`, `run_phase1_baselines.py`, `run_phase4_pilot.py`, `run_phase4_stage3_local.py`.
- **Input:** `--config` (required), `--results-dir` (optional), plus any caller-registered flags via `extra_args`.
- **Output:** Loaded config dict, a `RunContext`, and parsed `argparse.Namespace`.
- **Remarks:** **Extended this session** — added `extra_args` callback (Option 1) so callers needing custom flags can still get `RunContext` tracking. Return signature changed 2-tuple → 3-tuple; both existing callers updated. Verified live with a real config + 11 custom flags.

### `config/project_config.yaml`
- **Role:** Root project-wide settings (not phase-specific): model names, scope exclusions, `decision_window_ms: 50`, `sensor_channels`.
- **Called by:** Not directly loaded by any audited code path (informational/reference — cross-checked for consistency against other files instead).
- **Remarks:** Internally consistent with `inference.yaml` (model names) and `hcma_policy.yaml` (`decision_window_ms`). No issues.

---

## Phase 1 — Physics, Sensing, Baselines

### `src/simulation/sumo_env.py`
- **Role:** `PyBulletSumoEnv(gym.Env)` — the physics environment. One `step()` = one 50ms decision cycle.
- **Called by:** Every script that trains or evaluates anything — `run_phase1_baselines.py` (direct, via `DummyVecEnv`), `run_phase4_pilot.py`/`run_phase4_stage3_local.py` (direct), `watch_baseline.py`.
- **Input:** `EnvConfig`, `arena_config_path`, `opponent_policy: Callable[[obs], (left,right)]`. `step(action: np.ndarray[2])`.
- **Output:** `reset()`/`step()` return Gym-standard `(obs, reward, terminated, truncated, info)`. `obs` = `{"tof": ndarray(7), "ir": ndarray(2), "encoder": ndarray(4)}`.
- **Remarks:** **Fixed this session** (4 issues): duplicate `enable_reward_shaping` field; dead `EnvConfig.seed`; exact-float opponent-coast check; `SensorSuite` construction desynced from real chassis dimensions (root cause was in `sensors.py`'s consumption pattern, fixed at this call site). `_step_count` (used for truncation) is a plain Python int NOT captured by PyBullet's `saveState`/`restoreState` — `PyBulletSimulationBackend` (Phase 4) must snapshot/restore it separately, which it now does.

### `src/simulation/arena.py`
- **Role:** `Dohyo`/`DohyoSpec` — ring geometry construction, `is_outside()`/`has_fallen()`/`distance_to_edge()` boundary tests.
- **Called by:** `sumo_env.py` (`Dohyo.from_config()` in `reset()`).
- **Input:** `DohyoSpec.from_config(config_path)` reads `arena_config.yaml`'s `arena.*` block.
- **Output:** `radius_m`, `boundary_line_width_m`, `platform_top_z`; boolean tests take `(x,y,z)` position tuples.
- **Remarks:** **Fixed this session** — docstring falsely claimed "STUB, not implemented" despite being fully implemented. No functional bugs.

### `src/simulation/robot.py`
- **Role:** `RobotSpec`/`SumoRobot`/`build_robot()` — 4-wheel skid-steer chassis construction and motor control (`apply_pwm`, `coast`, `encoder_state`, `base_pose`, `base_velocity`).
- **Called by:** `sumo_env.py` (`build_robot()` in `reset()`), `rule_based_controller.py`/`ppo_controller.py` indirectly (via the env).
- **Input:** `RobotSpec.from_config()` reads `arena_config.yaml`'s `robot.*` block. `apply_pwm(left_pwm, right_pwm)` in `[-1, 1]`.
- **Output:** `SumoRobot` handle; `encoder_state()` returns `{left_pos, left_vel, right_pos, right_vel}`.
- **Remarks:** **Fixed this session** — duplicate imports, added a clarifying comment on the `chassis_mass_kg` config key vs. `total_mass_kg` field naming. **Unverified, flagged for empirical testing:** wheel spin-direction sign correctness can't be confirmed from static reading alone.

### `src/simulation/sensors.py`
- **Role:** `SensorSuite`/`SensorSpec` — simulated 3-channel sensor emission (ToF ray fan, IR edge probes, encoder readout).
- **Called by:** `sumo_env.py` (constructs one `SensorSuite` per robot in `reset()`).
- **Input:** `SensorSuite.read() -> {"tof": ndarray, "ir": ndarray, "encoder": ndarray}`.
- **Output:** Raw sensor arrays (no semantic interpretation — that's `lssd_encoder.py`'s job).
- **Remarks:** Root cause of the `sumo_env.py` desync bug (fixed at the call site). IR edge convention (high value = danger) correctly matches `rule_based_controller.py`'s and `lssd_encoder.py`'s thresholds.

### `src/preprocessing/ir_gradient.py`
- **Role:** `IRGradientFilter` — rolling-window rate-of-approach signal from raw IR readings.
- **Called by:** `perception_agent.py`, **and now `match_trainer.py`** (added this session — was previously missing entirely).
- **Input:** `update(ir_sample)` — one IR sample per call.
- **Output:** `float` — max rate-of-approach across probes.
- **Remarks:** Rate is NOT time-normalized (`buf[-1]-buf[0]`, no division by elapsed time) — flagged, not fixed, low practical impact (only affects the first ~4 calls after each `reset()`).

### `src/preprocessing/motor_deadband.py`
- **Role:** `MotorDeadbandFilter` — static per-sample threshold + differential-drive fusion into `{fwd, turn}`.
- **Called by:** `perception_agent.py`.
- **Input:** `apply(encoder_sample)` — `[left_vel, right_vel, left_pos, right_pos]`.
- **Output:** `{"fwd": float, "turn": float, "left_vel": float, "right_vel": float}`.
- **Remarks:** Clean, no issues. Correctly stateless (no `.reset()` needed or present).

### `src/preprocessing/savitzky_golay_filter.py`
- **Role:** `SavitzkyGolayFilter` — smooths the ToF channel over a rolling window.
- **Called by:** `perception_agent.py`.
- **Input:** `update(sample)` — multi-channel ToF sample.
- **Output:** `ndarray` — smoothed latest value per channel.
- **Remarks:** **Fixed this session** — `scipy` was missing from `requirements.txt` entirely; every script using `PerceptionAgent` would have crashed on first real smoothing call. Added `scipy>=1.11`.

### `src/baselines/rule_based_controller.py`
- **Role:** `RuleBasedController` — Baseline 1, hardcoded if-else state machine (edge-avoid > attack > search priority stack).
- **Called by:** `run_phase1_baselines.py` (as PPO's training opponent), `run_phase4_pilot.py`/`run_phase4_stage3_local.py` (as the default SBSO training opponent), `watch_baseline.py`.
- **Input:** `__call__(obs) -> (left_pwm, right_pwm)` — matches `OpponentPolicy` signature.
- **Output:** Normalized PWM pair.
- **Remarks:** **Enhanced this session** (separately, before this audit's Phase 1 pass) — search now covers ground (spin+creep) instead of standing still; attack now uses ray-drift lead prediction; added `RuleBasedParams.randomized()` for opponent-diversity training.

### `src/baselines/ppo_controller.py`
- **Role:** `PPOController`/`FlattenSumoObs` — Baseline 2 inference wrapper (loads a trained PPO model + VecNormalize stats).
- **Called by:** `watch_baseline.py`, `run_phase4_pilot.py`'s `_make_opponent_factory()`.
- **Input:** `PPOController.load(model_path, vec_normalize_path)`.
- **Output:** Callable opponent policy, same shape as rule-based.
- **Remarks:** Reviewed, no changes needed — pure inference wrapper, all training logic lives in `run_phase1_baselines.py`.

### `scripts/run_phase1_baselines.py`
- **Role:** Trains PPO (Baseline 2) against Baseline 1.
- **Called by:** You, directly, as a CLI entry point.
- **Input:** `--config` (via `build_run()`), e.g. `baseline2_ppo.yaml`.
- **Output:** Trained model + VecNormalize stats saved to `checkpoints/baseline2_ppo/`.
- **Remarks:** **Fixed this session** (two bugs) — opponent diversity (was training against one frozen fixed-parameter opponent all run) and a separate bug where `opponent.reset()` was never called between episodes at all. Both fixed via `RandomizedOpponentWrapper(gym.Wrapper)`.

### `scripts/watch_baseline.py`
- **Role:** Manual visual playback/debugging tool — watch any controller combination play.
- **Called by:** You, directly.
- **Input:** `--agent`/`--opponent` (`rule_based`/`ppo`/`idle`), `--ppo-model`, `--ppo-vecnorm`, etc.
- **Remarks:** Not modified. One CLI usage confusion resolved (merging two flags incorrectly). One harmless dead-code line noted (unused `ax` variable), not fixed.

### `config/arena_config.yaml`
- **Role:** Arena geometry, robot chassis spec, LSSD thresholds, episode timing, reward-shaping weights.
- **Consumed by:** `sumo_env.py`, `arena.py`, `robot.py`, `lssd_encoder.py`, `actuator_bridge.py`.
- **Remarks:** Audited — episode timing and reward-shaping keys match code exactly. Robot dynamics values (`max_wheel_speed_rad_s` etc.) differ from `RobotSpec`'s class-level defaults, but that's expected (config always wins). `lssd_thresholds` are explicitly marked PLACEHOLDER pending calibration.

---

## Phase 2 — Multi-Agent Reasoning Layer

### `src/agents/schemas.py`
- **Role:** All Pydantic data contracts — `MacroStrategy`, `OpponentBehavior`, `TacticalKeyword` enums; `PerceptionState`, `OpponentAnalysis`, `MacroStrategyDecision`, `TacticalCommand`, `SumoRobotState` models.
- **Called by:** Every Phase 2+ file — the shared vocabulary for the whole agent pipeline.
- **Remarks:** Clean, no issues. Key clarification: TEA emits a `TacticalKeyword` (categorical), NOT raw PWM directly — the Actuator Bridge does that mapping.

### `src/data/lssd_encoder.py`
- **Role:** `LSSDEncoder` — deterministic mapping from filtered sensor features to the compact LSSD text string (e.g. `"opp=near,dir=FC;edge=safe;mom=fwd"`).
- **Called by:** `perception_agent.py`, `match_trainer.py`, `run_phase4_pilot.py`.
- **Input:** `encode(tof_m, approach_rate, ego, detect_range_m=1.4) -> dict`.
- **Output:** `{"lssd_text": str, "opp_distance": str, "opp_direction": str, "edge": str, "momentum": str, "opp_distance_m": float, "edge_approach_rate": float}`.
- **Remarks:** Audited — fully compliant with `arena_config.yaml`'s `lssd_thresholds`, meters→cm unit conversion verified correct. Minor: 7-ray direction binning is a 3/2/2 split, not perfectly symmetric (cosmetic).

### `src/agents/perception_agent.py`
- **Role:** `PerceptionAgent` — orchestrates the full preprocessing chain (Savitzky-Golay + IR gradient + motor deadband) into one `LSSDEncoder.encode()` call.
- **Called by:** `graph_builder.py`'s `perception_node`, `run_phase4_stage2_local.py`, `run_phase4_stage3_local.py`.
- **Input:** `perceive(raw_tof, raw_ir, raw_encoder) -> PerceptionState`. **Stateful** — needs `.reset()` per episode/match.
- **Output:** A full `PerceptionState` Pydantic object.
- **Remarks:** Clean, correctly wired, no issues.

### `src/agents/opponent_analysis_agent.py`
- **Role:** `OpponentAnalysisAgent` (OAA) — SLM classification of opponent behavior from LSSD history.
- **Called by:** `graph_builder.py`'s `oaa_node`.
- **Input:** `analyze(lssd_history: list[str], frame_index: int) -> OpponentAnalysis`.
- **Output:** `OpponentAnalysis` with `behavior`, `confidence`, `frame_stamp`.
- **Remarks:** **Fixed this session** — `build_oaa_prompt()`'s rendered history order (newest-first) directly contradicted its own header text ("most recent last"). Now genuinely chronological, verified.

### `src/agents/strategy_agent.py`
- **Role:** `StrategyAgent` (SA) — SLM macro-strategy decision.
- **Called by:** `graph_builder.py`'s `sa_node`.
- **Input:** `decide(perception, prev_oaa) -> MacroStrategyDecision`.
- **Remarks:** **Fixed earlier this session (D5a)** — originally couldn't receive a DSPy-compiled prompt at all (predates Phase 4's DSPy compiler in the codebase's history). Now accepts/hot-swaps `prompt_program`.

### `src/agents/tactical_execution_agent.py`
- **Role:** `TacticalExecutionAgent` (TEA) — SLM conversion of macro-strategy + state into one `TacticalKeyword`.
- **Called by:** `graph_builder.py`'s `tea_node`.
- **Input:** `execute(perception, macro) -> TacticalCommand`.
- **Remarks:** Clean, no issues.

### `src/agents/actuator_bridge.py`
- **Role:** `ActuatorBridge` — deterministic `TacticalKeyword` → `(left_pwm, right_pwm)` mapping.
- **Called by:** `graph_builder.py`'s `bridge_node`.
- **Input:** `to_pwm(command: TacticalCommand) -> (float, float)`. `from_config()` reads `arena_config.yaml`'s `actuator_bridge.*` block (or defaults).
- **Remarks:** Clean, exhaustive keyword mapping, no issues.

### `src/agents/hardware_constraint_monitor.py`
- **Role:** `HCMAPolicy` — token-budget governor + emergency SA-bypass circuit-breaker logic.
- **Called by:** Nothing yet — `AgentDeps.hcma` field exists but no node function references it.
- **Input:** `compute_token_budget(agent_name, headroom_ratio)`, `should_bypass_sa(consumed_ratio)`.
- **Remarks:** Logic verified correct against `hcma_policy.yaml` (perfect key-path match). Confirmed **not wired into the live decision cycle** — matches the already-agreed Phase 5c-only scope, not a new problem.

### `src/agents/monolithic_reasoning_agent.py`
- **Role:** Intended: MRA, Baseline 3 (zero-shot monolithic single-call SLM).
- **Called by:** Nothing — genuinely unimplemented.
- **Remarks:** **Real, confirmed gap** (not stale docs like `arena.py` was) — this file is an actual stub (`raise NotImplementedError`). Will block Phase 5 evaluation of Baseline 3 until built.

### `src/orchestration/graph_builder.py`
- **Role:** The decision-cycle dataflow — pure node functions (`perception_node` → `oaa_node` → `sa_node` → `tea_node` → `bridge_node`) plus `build_langgraph()` to compile them into a real LangGraph `StateGraph`.
- **Called by:** `state.py`'s `SumoController`.
- **Remarks:** Clean, confirmed complete and correct (re-verified this session). Pipelined-staleness design (SA reads `prev_opponent_analysis`) implemented exactly as documented.

### `src/orchestration/state.py`
- **Role:** `SumoController` — the actual live-match orchestration driver. Holds rolling LSSD history + previous-frame OAA (the pipelining state), builds each frame's `SumoRobotState`, runs the decision cycle (pure sequence or compiled LangGraph), returns `(left_pwm, right_pwm)`.
- **Called by:** Nobody yet, per current audit — this is the live-match entry point that would be used by Phase 5 evaluation, not by Phase 4 training (which uses `MacroStrategyExecutor` instead, per design decision D2).
- **Input:** `decide(raw_tof, raw_ir, raw_encoder) -> (left_pwm, right_pwm, final_state)`. Call `.reset()` per episode.
- **Remarks:** Clean, well-designed. **Resolves an earlier misunderstanding of mine** — I'd previously assumed `state.py` might duplicate `schemas.py`'s `SumoRobotState`; it doesn't, it's the controller that uses it.

### `config/hcma_policy.yaml`
- **Role:** Token-budget governor + emergency-bypass policy values.
- **Consumed by:** `hardware_constraint_monitor.py`.
- **Remarks:** Perfect compliance, every key path matches exactly. Deferred features (`tiered_prompt_templates` etc.) correctly unread by any code.

---

## Cross-cutting bugs fixed this session (Phase 0-2 audit specifically)

1. `run_context.py` — dead timestamp-formatting code
2. `_script_common.py` — non-composable `build_run()`
3. `sumo_env.py` — 4 issues (duplicate field, dead seed, float-equality coast check, sensor/robot desync)
4. `arena.py` — stale stub docstring
5. `robot.py` — duplicate imports
6. `requirements.txt` — missing `scipy`
7. `opponent_analysis_agent.py` — history-order/prompt-text contradiction
8. `match_trainer.py` — `approach_rate` hardcoded to 0.0, meaning training data was always "edge=safe" regardless of actual danger (found while checking `lssd_encoder.py` compliance — the most significant single finding of this audit pass)

## Still open / needs your input
- `monolithic_reasoning_agent.py` — implement now or later?
- HCMA — wire into `graph_builder.py` now, or stay deferred to Phase 5c as planned?
- `ir_gradient.py`'s non-time-normalized rate + `lssd_thresholds`' self-acknowledged PLACEHOLDER status — real calibration pass needed at some point, not urgent
