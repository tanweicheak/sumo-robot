# Sumo-SBSO: End-to-End Technical Pipeline Report (Phase 0–4)

**Scope:** every file across Phase 0–4 plus cross-cutting infrastructure/logging.
Phase boundaries below follow each file's own declared `Phase:` docstring tag,
not an external categorization — this is the project's own source of truth.
Phase 5 (evaluation/statistics) is out of scope per this report's brief; it is
referenced only where a Phase 4 artifact hands off to it.

---

# PHASE 0 — Foundation, Configuration & Run Tracking

## 1. Architectural Role & Objective

Phase 0 has no robotics content of its own — it is the substrate every later
phase depends on to run reproducibly and to fail predictably instead of
silently. Three responsibilities: (a) load and merge YAML configuration with
an `extends:` inheritance chain, (b) validate that configuration against
per-phase Pydantic schemas before any expensive work starts, (c) stamp every
run with a reproducible identifier (git commit + config hash + timestamp) so
a result can always be traced back to exactly what produced it. Every
`scripts/run_*.py` entrypoint across every later phase calls into this layer
via `build_run()` before doing anything else. Downstream, it hands each script
a resolved `(config: dict, ctx: RunContext, args: argparse.Namespace)` triple.

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/common/config_loader.py` | `load_config()`, `require_keys()`, `validate_config()`, `ConfigError` | Recursive `extends:` merge of YAML configs; dotted-path required-key checks; dispatches to per-phase Pydantic validation | YAML file path (`str \| Path`) | `dict[str, Any]` (resolved, merged config) | `pyyaml`, `src.common.config_schemas` | **Complete, validated this session** — see Critical Remarks |
| `src/common/config_schemas.py` | `Phase4Config`, `Phase1Config`, `MCTSConfigSchema`, `DSPyRecompilationConfigSchema`, `PPOHyperparamsSchema`, `PHASE_SCHEMAS` registry | Pydantic type/range validation per phase; a cross-field validator flags DSPy `K` vs. self-checkpoint-interval misalignment | Resolved config `dict` | Raises `pydantic.ValidationError` (wrapped as `ConfigError`) or passes silently | `pydantic>=2.0` | **Complete** — registered phases: `phase4`, `phase4_pilot`, `phase1`. `phase4_stage3_local` intentionally NOT registered (CLI-driven config, no `mcts:`/`dspy_recompilation:` block) |
| `src/common/run_context.py` | `RunContext` dataclass, `_git_commit()`, `_config_hash()` | Builds `run_id = f"{variant}__{phase}__{timestamp}__{config_hash}"`; persists `run_context.json` | `variant_name`, `phase`, `resolved_config` | `RunContext` object; `.save()` writes JSON to `<results_dir>/<run_id>/run_context.json` | stdlib only (`hashlib`, `subprocess`, `datetime`) | **Complete** |
| `scripts/_script_common.py` | `build_run()`, `setup_logging()`, `poll_gpu_stats()` | Shared entrypoint every `scripts/run_*.py` calls: parses `--config`/`--results-dir` + caller-supplied extra CLI flags, loads+validates config, builds `RunContext`. Also hosts the cross-cutting logging and GPU-telemetry helpers (see Cross-Cutting section) | `phase: str`, `description: str`, `extra_args: Callable \| None` | `tuple[dict, RunContext, argparse.Namespace]` | `src.common.config_loader`, `src.common.run_context` | **Complete** |

## 3. Concrete Data Contracts & Interfaces

- **Config resolution**: any YAML file may declare `extends: <relative_path>.yaml`; `load_config()` resolves this recursively, child keys overriding parent keys via a deep dict merge (not a shallow overwrite — nested dicts merge key-by-key).
- **`RunContext.run_id` format**: `{variant_name}__{phase}__{YYYYMMDDTHHMMSSZ}__{12-char sha256 hex prefix of the sorted-key JSON config}`.
- **`build_run()` reserved CLI flags**: `--config` (required, str path), `--results-dir` (optional, str path). Any `extra_args` callback that tries to redefine either raises an `argparse` error at parse time, not silently.
- **`PHASE_SCHEMAS` keys are exact-match strings**, not prefixes — `"phase4"` does not match `"phase4_pilot"`. This is the exact mechanism behind Critical Remark #1 below.

## 4. Runtime Execution Lifecycle & State Management

1. Script starts → `build_run(phase=..., description=..., extra_args=...)` is the first call.
2. `argparse` parses `--config`, `--results-dir`, and any script-specific flags on one shared parser.
3. `load_config(args.config)` resolves the `extends:` chain into one flat dict.
4. `validate_config(config, phase)` looks up `phase` in `PHASE_SCHEMAS`; a registered phase gets full Pydantic validation (raises `ConfigError` and halts before any GPU/PyBullet work); an unregistered phase is a silent no-op (phased rollout, not all-or-nothing).
5. `RunContext(variant_name, phase, resolved_config)` is constructed — `run_id` and `config_hash` computed in `__post_init__`.
6. Caller receives `(config, ctx, args)` and proceeds with phase-specific logic.
7. No PyBullet/GPU state is touched anywhere in this phase — it is pure Python + filesystem + subprocess (for `git rev-parse`).

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **Exact-string phase matching is a real footgun** — already caused one confirmed bug this session (`run_phase4_pilot.py` using `phase="phase4_pilot"` while `PHASE_SCHEMAS` only had `"phase4"` registered, silently skipping validation on the actual RunPod entrypoint). Fixed, but any *future* script added under a new phase string must be explicitly registered or it silently gets zero validation — no warning, no error.
- **`git rev-parse HEAD` failure is swallowed to `"unknown"`** — on a RunPod pod without `.git` present (e.g. code copied via `scp`/tarball rather than `git clone`), every `run_id` silently loses commit traceability with no error raised. Worth confirming the RunPod deployment method preserves `.git/`.
- **`Phase4Config`'s `model_config = {"extra": "allow"}`** at the top level, but nested schemas (`MCTSConfigSchema`, `DSPyRecompilationConfigSchema`) inherit Pydantic v2's default `extra='ignore'` — undeclared nested keys (e.g. `mcts.decision_cycles`, present in real configs but not in the schema) are silently dropped from validation, not rejected. This is intentional (avoids over-fitting the schema to every config field) but means the schema validates *less* than the full config surface.

---

# PHASE 1 — Physics Simulation, Arena, Sensors & Non-Learned/RL Baselines

## 1. Architectural Role & Objective

Phase 1 is the ground-truth physical world every later phase's decisions are
ultimately executed in, plus the two baselines (rule-based, PPO) that require
no SLM at all. Upstream, it depends only on Phase 0's config system. Downstream,
`PyBulletSumoEnv` is the single simulation surface every later phase drives:
Phase 2's agents read its sensor output, Phase 4's MCTS snapshots/restores its
internal PyBullet state directly, and Phase 5 (out of scope) will replay real
sensor traces against the same `arena_config.yaml`-defined geometry.

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/simulation/arena.py` | `DohyoSpec`, `Dohyo`, `.is_outside()`, `.has_fallen()`, `.has_capsized()`, `.distance_to_edge()` | Builds the ring platform in PyBullet; three independent terminal-condition checks (radial edge, z-drop, orientation) | `(x,y,z)` position tuples, `(x,y,z,w)` orientation quaternions, `config/arena_config.yaml` geometry | `bool` per check; `float` for distance-to-edge | `pybullet` | **Complete, extended this session** (capsize check added) |
| `src/simulation/robot.py` | `RobotSpec`, `SumoRobot`, `build_robot()`, `.apply_pwm()`, `.base_pose()`, `.encoder_state()` | Constructs the 4-wheel differential-drive chassis; translates `(left_pwm, right_pwm) ∈ [-1,1]` into `VELOCITY_CONTROL` joint targets | `left_pwm: float`, `right_pwm: float` | `base_pose() -> (pos, orn)`; `encoder_state() -> dict[left_vel,right_vel,left_pos,right_pos]` | `pybullet` | **Complete, wheel-direction sign fixed this session** (empirically verified) |
| `src/simulation/sensors.py` | `SensorSpec`, `SensorSuite`, `.read()` | Emits exactly 3 raw sensor channels per robot: ToF ray fan, IR edge probes, motor encoder — no IMU, no semantic interpretation | Robot pose (internal), `SensorSpec` geometry | `dict[str, np.ndarray]`: `"tof"` shape `[7]` float32, `"ir"` shape `[2]` float32, `"encoder"` shape `[4]` float32 | `pybullet` (`rayTestBatch`), `numpy` | **Complete** |
| `src/simulation/sumo_env.py` | `EnvConfig`, `PyBulletSumoEnv`, `.reset()`, `.step()`, `._compute_reward()` | Gymnasium-style env wrapping arena+robot+sensors; computes terminal outcome and reward | `action: np.ndarray[2]` (left/right PWM) | `(obs, reward, terminated, truncated, info)`; `info` includes `outcome`, `agent_out_reason`, `opponent_out_reason`, `agent_pos`, `opponent_pos` | `gymnasium`, `pybullet`, `src.simulation.{arena,robot,sensors}` | **Complete, extended this session** (capsize wiring, outcome-reason fields) |
| `src/baselines/rule_based_controller.py` | `RuleBasedParams`, `RuleBasedController`, `make_rule_based_policy()`, `make_randomized_opponent_factory()` | Baseline 1 — deterministic priority-stack state machine (edge-avoid > attack > search); `.randomized()` samples parameter jitter for opponent diversity | `obs: dict[str, np.ndarray]` (tof/ir) | `(left_pwm, right_pwm): tuple[float, float]` | `numpy` | **Complete** |
| `src/baselines/ppo_controller.py` | `PPOController`, `FlattenSumoObs`, `.load()` | Baseline 2 — Stable-Baselines3 PPO wrapper; flattens the dict observation for SB3's `Box` space requirement | Flattened obs vector | `(left_pwm, right_pwm)` at inference; SB3 `PPO` model artifacts at training | `stable_baselines3`, `torch` | **Complete** |
| `src/baselines/adversarial_bait_controller.py` | `BaitControllerParams`, `BaitController`, `make_bait_controller_policy()` | Diagnostic-only out-of-family opponent (lure→counter→patrol→edge_avoid state machine); NOT one of the 5 official evaluation conditions | `obs: dict[str, np.ndarray]` | `(left_pwm, right_pwm)` | `numpy` | **Complete, built this session; iterated 3x on real evidence** (see session record §2) |
| `scripts/run_phase1_baselines.py` | `main()` | Trains Baseline 2 (PPO) against randomized Baseline 1; evaluates both | `config/baseline2_ppo*.yaml` | Saved SB3 model + `VecNormalize` stats; win/loss/draw eval summary | `stable_baselines3`, `src.baselines.*` | **Complete, wandb + structured logging added this session** |

## 3. Concrete Data Contracts & Interfaces

- **`SensorSuite.read()` output shapes** — `tof: [7]` float32 (0.0–1.5m range, `tof_max_range_m` on no-hit), `ir: [2]` float32 (0.0–1.0 normalized ray fraction; `1.0` = "over the edge, no floor hit"), `encoder: [4]` float32 = `[left_vel, right_vel, left_pos, right_pos]`.
- **IR probe mounting is deliberately BELOW the chassis bottom face** (`mount_drop ≈ -0.045`) — casting from above hits the robot's own hull first and the self-hit filter discards it, leaving no platform hit at all. This is a documented, load-bearing geometry detail, not arbitrary.
- **Action space**: `np.ndarray[2]`, `left_pwm, right_pwm ∈ [-1.0, 1.0]`.
- **Terminal outcome values**: `info["outcome"] ∈ {"win", "loss", "draw"}`; `info["agent_out_reason"] / info["opponent_out_reason"] ∈ {"pushed_out", "fell_off_edge", "capsized", None}`.
- **`arena_config.yaml` keys read**: `radius_m`, `platform_top_z`, chassis dimensions matching `RobotSpec`, `reward_shaping.{fwd_weight,push_weight,edge_bonus,track_weight,glance_penalty,spin_penalty,selfeject_penalty,time_penalty}`, `episode.max_episode_seconds`, `lssd_thresholds.*` (consumed by Phase 2, defined here).
- **`control_dt_s = 0.05`**, `substeps = round(control_dt_s / (1/240)) = 12` — fixed, verified-consistent physics/control timestep relationship.

## 4. Runtime Execution Lifecycle & State Management

1. `PyBulletSumoEnv.__init__`: connects `p.DIRECT` (headless) unless `use_gui=True`; builds `Dohyo` platform once.
2. `.reset()`: respawns both robots at fixed offsets (`±spawn_offset_m, 0.0`), resets `RuleBasedController`/`BaitController` internal state machines via their own `.reset()`, zeroes `_step_count`.
3. `.step(action)`: applies agent PWM, steps `opponent_policy(opp_obs)` for the opponent, advances physics `substeps` times, computes `is_outside/has_fallen/has_capsized` for both robots, derives `outcome` and `reward`, increments `_step_count`, checks `_step_count >= max_steps` for `truncated`.
4. **No PyBullet state snapshotting occurs in Phase 1** — `saveState`/`restoreState` is exclusively a Phase 4 (MCTS) concept; Phase 1's env always advances forward, never rewinds.
5. `RuleBasedController`'s internal state machine (edge-avoid reverse/pivot phases, search spin/creep alternation, attack commit-latch) persists across `.step()` calls within an episode via instance attributes (`_edge_phase`, `_search_timer`, etc.), cleared only by `.reset()`.

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **Headless safety**: `use_gui=False` is the only mode safe for RunPod (no X server). Every training/eval script defaults to this; `--gui` is explicitly local-only, confirmed by every script that exposes it.
- **Capsize is a real, measured, non-zero risk**: even after this session's fixes, `rule_based` self-capsized 6/500 times in stress testing. Root cause (physical instability vs. behavior-specific) not yet isolated — flagged, not fixed.
- **`rule_based`'s commit-charge (`commit_cycles=10`, uninterrupted) structurally advantages it in any sustained push contest** — directly relevant to PPO training quality, since PPO trains against this exact behavior (see current open discussion in `session_record.md`).
- **Sensor dropout is not simulated** — `SensorSuite.read()` always returns a full, valid reading in simulation; real hardware sensor dropout/noise is exclusively a Phase 5c (HEL, real sensor traces) concern, not modeled here.

---

# PHASE 2 — Multi-Agent Perception, Reasoning & Orchestration Architecture

## 1. Architectural Role & Objective

Phase 2 is the reasoning stack: it turns raw Phase 1 sensor arrays into
semantic, LLM-consumable state text, and defines the four-agent decision
pipeline (PA → {OAA, SA} → TEA → Actuator Bridge) plus its LangGraph
orchestration. Every agent here is built against `SLMClient`, an abstract
interface — Phase 2 itself only ever exercises the `MockSLMClient`; Phase 3
supplies the real backends without any agent code changing. Downstream, the
same agent classes and the same `SumoRobotState` schema are reused verbatim
in Phase 4's live-match execution (`_execute_strategy_live`) and will be
reused again in Phase 5's evaluation match-runner (not yet built).

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/agents/schemas.py` | `OpponentBehavior`, `MacroStrategy`, `TacticalKeyword`, `DistanceLabel`, `DirectionLabel`, `EdgeLabel`, `MomentumLabel` (all `str, Enum`); `PerceptionState`, `OpponentAnalysis`, `MacroStrategyDecision`, `TacticalCommand`, `SumoRobotState` (all `pydantic.BaseModel`) | Canonical data contracts for the entire multi-agent pipeline and the LangGraph state object | N/A (schema definitions) | N/A | `pydantic>=2.0` | **Complete** |
| `src/preprocessing/savitzky_golay_filter.py` | `SavitzkyGolayFilter`, `.update()`, `.reset()` | Smooths the ToF channel via `scipy.signal.savgol_filter`; passes raw samples through until the window fills | `sample: array-like`, per-channel | `np.ndarray[n_channels]` float32 | `scipy>=1.11`, `numpy` | **Complete** |
| `src/preprocessing/ir_gradient.py` | `IRGradientFilter`, `.update()`, `.reset()` | Rolling-window rate-of-approach signal from the IR channel, normalized to a true per-second rate | `ir_sample: array-like`, `dt_s: float` | `float` (units/sec) | `numpy` | **Complete, units bug fixed this session** (was returning an un-normalized raw window delta) |
| `src/preprocessing/motor_deadband.py` | `MotorDeadbandFilter`, `.apply()` | Deadband-filters encoder velocity noise; fuses left/right wheel velocity into `(fwd, turn)` ego-motion | `encoder_sample: [left_vel,right_vel,left_pos,right_pos]` | `dict[fwd,turn,left_vel,right_vel]` | `numpy` | **Complete** |
| `src/data/lssd_encoder.py` | `LSSDEncoder`, `.encode()`, `.from_config()` | Deterministic mapping from filtered features to the compact LSSD key-value string vocabulary | `tof_m: array`, `approach_rate: float`, `ego: dict` | `dict` incl. `"lssd_text": str` (e.g. `"opp=near,dir=FC;edge=safe;mom=fwd"`) | `numpy`, `src.common.config_loader` | **Complete** |
| `src/agents/perception_agent.py` | `PerceptionAgent`, `.perceive()`, `.reset()` | Orchestrates SG filter + IR gradient + LSSD encoding into one `PerceptionState` | Raw `tof`, `ir`, `encoder` arrays | `PerceptionState` (pydantic) | `src.preprocessing.*`, `src.data.lssd_encoder` | **Complete, `control_dt_s` param added this session** |
| `src/agents/opponent_analysis_agent.py` | `OpponentAnalysisAgent`, `build_oaa_prompt()`, `.analyze()` | Classifies opponent behavior from a chronological LSSD history window; stamps `frame_stamp` for the pipelined-staleness design | `lssd_history: list[str]`, `frame_index: int` | `OpponentAnalysis` (pydantic) | `src.inference.slm_client.SLMClient` | **Complete** |
| `src/agents/strategy_agent.py` | `StrategyAgent`, `build_sa_prompt()`, `.decide()`, `.update_prompt_program()` | Selects a `MacroStrategy` from current perception + the PREVIOUS frame's OAA (pipelined staleness); prompt is DSPy-compiled-program-aware | `PerceptionState`, `prev_oaa: OpponentAnalysis \| None` | `MacroStrategyDecision` (pydantic) | `src.inference.slm_client.SLMClient` | **Complete** |
| `src/agents/tactical_execution_agent.py` | `TacticalExecutionAgent`, `build_tea_prompt()`, `.execute()` | Converts macro-strategy + state into one `TacticalKeyword` | `PerceptionState`, `MacroStrategyDecision` | `TacticalCommand` (pydantic, single enum field) | `src.inference.slm_client.SLMClient` | **Complete** |
| `src/agents/monolithic_reasoning_agent.py` | `MonolithicReasoningAgent`, `build_mra_prompt()`, `.decide()` | Baseline 3 — one direct call, state→keyword, no decomposition. Deliberately reuses the same `TacticalCommand`/`ActuatorBridge` as the multi-agent path | `PerceptionState` | `TacticalCommand` | `src.inference.slm_client.SLMClient` | **Complete** (agent only — standalone eval loop wiring is separate follow-up work, not yet built) |
| `src/agents/actuator_bridge.py` | `ActuatorBridge`, `.to_pwm()`, `.from_config()` | Deterministic, non-SLM mapping from `TacticalKeyword` to `(left_pwm, right_pwm)` | `TacticalCommand` | `tuple[float, float]` | `src.common.config_loader` | **Complete** |
| `src/agents/hardware_constraint_monitor.py` | `HCMAPolicy`, `.compute_token_budget()`, `.should_bypass_sa()` | Deterministic, non-SLM token-budget governor + emergency SA-bypass circuit breaker | `agent_name: str`, `headroom_ratio: float`; `consumed_ratio: float` | `int` (max_new_tokens, or `-1`=uncapped); `bool` (bypass decision) | `src.common.config_loader` (`config/hcma_policy.yaml`) | **Complete.** Latency feed is MOCKED in Phase 2, real in Phase 3+ |
| `src/orchestration/graph_builder.py` | `AgentDeps`, `perception_node`/`oaa_node`/`sa_node`/`tea_node`/`bridge_node`, `NODE_SEQUENCE`, `build_langgraph()` | Pure node functions over `SumoRobotState`, each self-timed; assembles the real LangGraph `StateGraph` or exposes `NODE_SEQUENCE` for a LangGraph-free dev path | `SumoRobotState`, `AgentDeps` | `dict` of state updates per node; compiled `StateGraph` from `build_langgraph()` | `langgraph` (only for `build_langgraph`) | **Complete** |
| `src/orchestration/state.py` | `SumoController`, `.decide()`, `.reset()` | Drives the frame loop: builds each frame's `SumoRobotState`, runs either `NODE_SEQUENCE` or the compiled graph, rolls the pipelined OAA store forward | `raw_tof`, `raw_ir`, `raw_encoder` per frame | `(left_pwm, right_pwm, final_state: SumoRobotState)` | `src.orchestration.graph_builder` | **Complete** |

## 3. Concrete Data Contracts & Interfaces

- **LSSD text grammar**: `"opp={near|mid|far|none},dir={FL|FC|FR|none};edge={safe|warning|critical};mom={fwd|rev|turn|still}"` — a fixed, deterministic key-value string, never freeform text.
- **`PerceptionState` fields**: `lssd_text: str`, `opp_distance: DistanceLabel`, `opp_direction: DirectionLabel`, `edge: EdgeLabel`, `momentum: MomentumLabel`, `opp_distance_m: float`, `edge_approach_rate: float`.
- **`MacroStrategy` vocabulary** (5 values): `charge`, `flank`, `retreat`, `hold`, `evade_edge`.
- **`TacticalKeyword` vocabulary** (7 values): `charge_forward`, `arc_left`, `arc_right`, `pivot_left`, `pivot_right`, `reverse`, `stop`.
- **`OpponentBehavior` vocabulary** (4 values): `aggressive`, `evasive`, `defensive`, `unknown`.
- **`SumoRobotState`** (the LangGraph state object): `frame_index: int`, `raw_tof/raw_ir/raw_encoder: list[float]`, `lssd_history: list[str]`, `perception/opponent_analysis/prev_opponent_analysis/macro_strategy/tactical_command: Optional[...]`, `left_pwm/right_pwm: float`, `timing_ms: dict[str, float]`.
- **`lssd_thresholds` config keys** (`arena_config.yaml`, still-placeholder values): `distance_cm.{near_max, mid_max}`, `edge_approach_rate.{stable_max, approaching_max}`.
- **`hcma_policy.yaml` keys**: `decision_window_ms`, `token_budget_governor.{enabled, base_max_new_tokens, min_max_new_tokens, headroom_scaling.{full_budget_above_ratio, tighten_below_ratio}}`, `emergency_bypass.{enabled, trigger_consumed_ratio}`.

## 4. Runtime Execution Lifecycle & State Management

1. `SumoController.__init__` builds `AgentDeps` (bundles PA/OAA/SA/TEA/bridge/HCMA instances); if `use_langgraph=True`, compiles the graph once via `build_langgraph()`.
2. **Per-episode**: `.reset()` clears the PA's internal filter buffers, the `deque(maxlen=history_len)` LSSD history, `_prev_oaa`, and `_frame` counter.
3. **Per-frame `.decide()`**: builds a fresh `SumoRobotState` carrying `prev_opponent_analysis` (frame t-1's OAA — the pipelined-staleness mechanism) and the rolling `lssd_history`.
4. Executes either `NODE_SEQUENCE` sequentially (5 pure function calls, each updating state via `model_copy`) or the compiled LangGraph (`graph.invoke(state)`, normalized back to a `SumoRobotState` since LangGraph may return a plain dict depending on version).
5. **State roll-forward**: `state.opponent_analysis` (this frame's OAA, computed for use NEXT frame) becomes `self._prev_oaa`; the new `perception.lssd_text` is appended to the history deque; `_frame` increments.
6. **No PyBullet state is touched in Phase 2 at all** — this entire phase operates purely on the `SumoRobotState` Pydantic object and the abstract `SLMClient` interface; it has no direct physics dependency.
7. **Concurrency**: `NODE_SEQUENCE`/LangGraph execution here is strictly linear (perception→oaa→sa→tea→bridge, one node at a time) — the pipelined *data* dependency (SA reads t-1's OAA, not t's) is what *permits* OAA/SA to run concurrently as a Phase 3 optimization, but Phase 2 itself does not implement that concurrency; it only creates the data-dependency structure that makes it safe to add later.

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **Pipelined staleness is a deliberate, documented one-frame lag**, not a bug: SA always reasons over an OAA classification that is one frame old. If this project's timing budget is ever tightened further, verify no code path accidentally "fixes" this into same-frame synchronous coupling, which would silently change the latency profile the pipelining was designed to improve.
- **`lssd_thresholds` are still placeholder values**, and — following this session's `ir_gradient.py` fix — the units feeding `edge_approach_rate` changed scale (~5x) without a corresponding recalibration of `stable_max`/`approaching_max`. Any Phase 2 output text generated before vs. after that fix is not directly comparable.
- **`LangGraph`'s return type is version-dependent** (`dict` vs. a model instance) — `SumoController.decide()` already defensively normalizes this, but any future LangGraph version upgrade should re-verify this branch still triggers correctly.
- **`MonolithicReasoningAgent` has no standalone execution loop** — it exists as an agent class only; wiring it into a runnable eval script is explicitly documented as separate, not-yet-done follow-up work. Do not assume Baseline 3 is runnable end-to-end today.

---

# PHASE 3 — Real SLM Inference Backends, Constrained Decoding & Hardware Emulation

## 1. Architectural Role & Objective

Phase 3 replaces Phase 2's `MockSLMClient` with real, constrained-decoding
inference backends, without any agent code changing (the whole point of the
`SLMClient` abstraction). It also introduces the Hardware Emulation Layer
(HEL) — the mechanism that will later, in Phase 5c only, throttle CPU/RAM and
inject latency to emulate the target Jetson Nano deployment device. Upstream,
it depends on Phase 2's agent/schema contracts. Downstream, Phase 4's MCTS
and live-match execution consume these same backends directly (`LlamaCppJudge`
locally, `SGLangJudge`/`RealDSPyCompiler` on RunPod).

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/inference/slm_client.py` | `SLMClient` (ABC), `MockSLMClient`, `ScriptedSLMClient` | Abstract backend contract; mock returns schema-valid synthesized instances (enum fields via random choice) for dev/CI; `ScriptedSLMClient` returns fixed, caller-specified responses for deterministic tests | `prompt: str`, `schema: Type[BaseModel]` | `BaseModel` instance of `schema` | `pydantic` | **Complete** |
| `src/inference/grammar.py` | `enum_choice_grammar()`, `enum_regex_pattern()`, `primary_enum_field()` | Builds GBNF (llama.cpp) and regex (SGLang) constraints forcing single-token enum output; longest-value-first ordering prevents short-value prefix collisions | `values: list[str]` | GBNF string / regex string; `primary_enum_field` returns `(field_name, enum_class)` | stdlib `re` only | **Complete** |
| `src/inference/llama_cpp_server.py` | `LlamaCppSLMClient`, `._ensure_loaded()`, `._grammar_for()` | Real backend via `llama-cpp-python`, in-process, GBNF-constrained, greedy decoding, grammar caching by enum class name, prompt-prefix KV-cache reuse (constant system prompts) | `prompt: str`, `schema: Type[BaseModel]` | Schema instance with the single enum field filled from constrained output | `llama-cpp-python>=0.2.90` | **Complete.** No batching — sequential calls only |
| `src/inference/sglang_server.py` | `SGLangSLMClient`, `.generate_structured_batch()` | Real backend via SGLang HTTP server; concurrent dispatch via `ThreadPoolExecutor` so server-side continuous batching engages; `threading.Lock()` protects `call_count` under concurrent access | `prompt/schema` (single) or `list[(prompt, schema)]` (batch) | Schema instance(s), order-preserved for batch | `requests` (client-side); SGLang server (separate process) | **Complete, thread-safe** — this is the version actually imported everywhere (`factory.py`, `test_sglang_server.py`) |
| `src/inference/sglang_client.py` | `SGLangSLMClient` (duplicate class, same name) | **Byte-for-byte earlier duplicate of `sglang_server.py` MINUS the `threading.Lock()` fix** | (identical to above) | (identical to above) | `requests` | **Orphaned dead code — nothing imports this file.** See Critical Remarks |
| `src/inference/factory.py` | `build_slm_client()` | Config-driven backend selection (`mock` \| `llama_cpp` \| `sglang`) — the single construction point every real script should use | `config/inference.yaml` path | `SLMClient` instance (concrete subclass per `backend:` key) | `src.common.config_loader`, all three backend modules | **Complete** |
| `src/hel/emulation_profile.py` | `HELProfile`, `.from_config()`, `.inject_latency_ms()` | Loads a device constraint profile (e.g. Jetson Nano); latency injection is a no-op unless explicitly enabled — Phase 5a/5b pass measured latency through unchanged by design | `config/jetson_nano.yaml` | `float` (possibly-inflated latency) | `src.common.config_loader` | **Complete.** Active only when Phase 5c enables it |
| `src/hel/hel_logger.py` | `HELLogRecord`, `HELLogger`, `.log_cycle()`, `.latency_compliance_rate()`, `.mean_latency_ms()` | Per-cycle timing/deadline-compliance recording from `SumoRobotState.timing_ms`; computes the KPIs used across all of 5a/5b/5c | `frame_index: int`, `timing_ms: dict`, optional `injected_ms` | `HELLogRecord`; `float` compliance rate / mean latency | stdlib only | **Complete** |
| `src/evaluation/latency_profiler.py` | (stub) | HEL-integrated per-cycle latency capture for Phase 5 | — | — | stdlib only (deliberately no heavy imports at module level) | **STUB — Phase 0 scaffold, not implemented** |
| `src/evaluation/schema_validity_checker.py` | (stub) | TEA schema-validity rate under clean vs. noisy input, for Phase 5 | — | — | stdlib only | **STUB — Phase 0 scaffold, not implemented** |

## 3. Concrete Data Contracts & Interfaces

- **`inference.yaml backend:` values**: `mock` \| `llama_cpp` \| `sglang`.
- **`llama_cpp:` config keys**: `model_path`, `n_ctx` (default 2048), `n_gpu_layers` (`-1` = full Metal/GPU offload), `temperature` (0.0 — greedy), `max_tokens` (default 8), `seed`, `verbose`.
- **`sglang:` config keys**: `agent_server_url`, `judge_server_url`, `temperature`, `max_tokens`, `timeout_s`, `max_concurrency`; nested `launch:` block consumed by `launch_sglang_servers.sh` (`agent_model_path`, `agent_port`, `judge_model_path`, `judge_port`, `*_mem_fraction_static`, `host`, `startup_timeout_s`).
- **Constrained-decoding contract**: every real SLM call emits **exactly one categorical token** matching an enum value — never a JSON object. `primary_enum_field()` finds the first `Enum`-typed field on any schema and that is the only field the model generates; every other field (e.g. `OpponentAnalysis.frame_stamp`) is stamped by calling code, not the model.
- **`hcma_policy.yaml`** decision window default: `50ms`.

## 4. Runtime Execution Lifecycle & State Management

1. `build_slm_client()` reads `backend:` once and returns a concrete client — no runtime backend switching within a run.
2. **`LlamaCppSLMClient`**: model loads lazily on first call (`_ensure_loaded`); grammar objects are cached per enum class name so repeated calls for the same schema type reuse the compiled grammar; each call is a single blocking `llm.create_completion()`.
3. **`SGLangSLMClient`**: stateless HTTP client; `generate_structured_batch()` fans out N requests across a `ThreadPoolExecutor` sized to `min(max_concurrency, len(requests))`, each thread doing its own blocking `requests.post()`; results collected in original order via a pre-sized `results` list indexed by submission order, not completion order.
4. **HEL**: `HELProfile` is constructed unconditionally (cheap), but `inject_latency_ms()` is a pure pass-through unless `latency_injection_enabled` is set — meaning full-power Phase 4/5a/5b runs carry zero HEL overhead by construction, not by a runtime flag check that could be forgotten.
5. **No PyBullet state interaction** in this phase — it is pure inference-serving infrastructure.
6. **Concurrency pattern**: thread-based (Python `ThreadPoolExecutor`), not process-based or `asyncio` — appropriate here because the actual work (HTTP I/O) releases the GIL, so true parallelism is achieved despite Python's GIL.

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **`sglang_client.py` is a confirmed orphaned duplicate** of `sglang_server.py`, missing the `threading.Lock()` fix around `call_count`. Nothing currently imports it (`factory.py` and `test_sglang_server.py` both import from `sglang_server.py`), so it is not corrupting anything today — but it is a live landmine: any future code (or a well-meaning future refactor) that imports from the more "naturally named" `sglang_client` module instead would silently reintroduce the race condition. **Recommend deleting this file, not fixing it** — it duplicates, does not extend, `sglang_server.py`.
- **`LlamaCppJudge`/`LlamaCppSLMClient` have zero batching support** — every candidate branch/position scored during MCTS pruning is a fully sequential local call. Confirmed this session: at `sim_budget=15` with pruning active, this is ~95 sequential 8B-model calls per single decision — tens of minutes to hours per episode on CPU/MPS. This is why `SGLangJudge` (concurrent) is the only viable backend for any real-scale RunPod run; `LlamaCppJudge` should be treated as local-smoke-test-only, never a real-scale path.
- **VRAM**: `inference.yaml`'s `launch.*_mem_fraction_static: 0.45` for both agent and judge servers sums to 0.90 of GPU memory — leaves ~10% headroom for CUDA context, PyTorch/PyBullet overhead, and any Phase 4 GPTQ quantization run sharing the same GPU. If both SGLang servers are launched AND a GPTQ quantization pass is attempted concurrently on the same GPU, this headroom is almost certainly insufficient — GPTQ should run only after stopping (or before starting) the SGLang servers, not alongside them.
- **`n_gpu_layers=-1` for local `llama_cpp`** assumes a Metal-capable `llama-cpp-python` build; a plain `pip install llama-cpp-python` on macOS often silently produces a CPU-only build, which would explain otherwise-unexplained extreme slowness (flagged this session as a likely contributor to the sim_budget sweep's slowness) — worth confirming build flags before attributing all slowness to the algorithm.
- **Grammar/regex correctness is load-bearing and asymmetric between backends**: `enum_choice_grammar` (GBNF) and `enum_regex_pattern` (regex) are two independently-written constraints for the same guarantee — a bug introduced in one would not be caught by tests of the other. `test_grammar.py`'s coverage (per earlier audit) only validates grammar-string construction, not real decode-time enforcement under load for either backend.

---

# PHASE 4 — Simulation-Based Strategy Optimization (SBSO), Training & Model Export

## 1. Architectural Role & Objective

Phase 4 is the core research contribution: it turns Phase 1–3's simulation and
inference infrastructure into a closed-loop training pipeline that produces
Benchmark 2 (and its 4 ablations) — MCTS strategy search over a scripted proxy
opponent, Judge-scored branch pruning, DSPy prompt recompilation, and LoRA
fine-tuning, followed by a two-branch export (GPTQ INT4 for SGLang cloud
serving, llama.cpp Q4_K_M for edge/HEL deployment). Upstream, it depends on
Phase 1's `PyBulletSumoEnv` (via `saveState`/`restoreState` snapshotting,
unique to this phase) and Phase 2/3's agent+inference stack (both directly,
via `StrategyAgent`, and indirectly, via the scripted `MacroStrategyExecutor`
proxy used during search). Downstream, its artifacts (`training_pairs.jsonl`,
LoRA adapters, quantized GGUF/GPTQ models) are exactly what Phase 5 (out of
scope) will evaluate.

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/sbso/mcts.py` | `MCTSNode`, `MCTSResult`, `MCTS`, `.search()`, `._legal()`, `._select()`, `._expand()`, `._backprop()` | Standard UCT tree search (`c_uct=1.41` default, never calibrated) over a `SimulationBackend`; Judge-based pre-pruning in `_legal()` when `judge_prune_threshold>0` | `root_state: Any`, `strategies: list[MacroStrategy]` | `MCTSResult` incl. `best_strategy`, `root_stats` (per-strategy mean value), `training_pair` | `src.sbso.simulation_backend` | **Complete.** `sim_budget`/`c_uct` never calibrated — sweep tooling built this session, not yet run at scale |
| `src/sbso/judge.py` | `Judge` (ABC), `MockJudge`, `LlamaCppJudge`, `SGLangJudge`, `.score_position()`, `.score_branch()`, `.score_branches_batch()` | Scores a position or candidate branch 1–5→`[0,1]`; only `SGLangJudge` implements concurrent batch scoring | `state_desc: str`, `strategy: MacroStrategy` | `float ∈ [0,1]` (single) or `list[float]` (batch) | `llama-cpp-python` or `requests`, per backend | **Complete.** Never validated against real match outcomes — diagnostic built this session, not yet run |
| `src/sbso/dspy_compiler.py` | `DSPyCompiler` (ABC), `MockDSPyCompiler`, `RealDSPyCompiler`, `.compile()`, `validate_prompt_candidate()` | Compiles a new few-shot `prompt_program` string from accumulated `training_pairs`; `validate_prompt_candidate()` (new) state-replay-validates a candidate against the incumbent before commit | `training_pairs: list`, `prompt_program: str` (incumbent) | New `prompt_program: str`; `validate_prompt_candidate()` returns `dict[accept, incumbent_mean_score, candidate_mean_score]` | `dspy-ai==3.3.0` (Real only) | **Complete.** `RealDSPyCompiler` never exercised against a live server yet — first real test will be the pilot run |
| `src/sbso/recompilation_scheduler.py` | `RecompilationScheduler`, `.should_recompile()` | Trigger (a) every K episodes, trigger (b) rolling-win-rate drop > delta vs. prior window | `episode: int`, `rolling_winrate: float \| None` | `tuple[bool, str \| None]` (`"k_episodes"` \| `"reward_drop"` \| `None`) | stdlib only | **Complete.** Documented intended defaults `K=500`/`W=100`; `_shared_defaults.yaml` had `K=5` (100x off, user-confirmed fixed to 500), `W=10` still open |
| `src/sbso/opponent_pool.py` | `OpponentDescriptor`, `OpponentPool`, `.sample()`, `._weighted_choice()` | Samples training opponent type (`baseline1`/`baseline2`/`self_checkpoint`) per episode against a warmup-then-target distribution | `episode: int`, `has_self_checkpoint: bool` | `OpponentDescriptor` (subclass of `str`) | stdlib only | **Complete, heavily fixed in prior sessions** (per handover doc) |
| `src/sbso/self_checkpoint_manager.py` | `SelfCheckpointManager`, `.maybe_snapshot()`, `.sample()` | Snapshots `(episode, prompt_program, dominant_strategy)` every `interval` episodes; samples uniformly across ALL accumulated checkpoints | `episode: int`, `prompt_program: str`, `dominant_strategy: str \| None` | `bool` (snapshot taken); `tuple[int,str,str\|None] \| None` (sample) | stdlib only | **Complete, report §3.2.1.3 validated against this code directly this session — matches exactly** |
| `src/sbso/simulation_backend.py` | `SimulationBackend` (ABC), `MockSimulationBackend`, `PyBulletMCTSState`, `PyBulletSimulationBackend`, `.root_state()`, `.step()`, `.rollout()`, `.judge_branch()`, `._restore()`, `._snapshot()` | The MCTS↔physics bridge: every `step()`/`rollout()` call during search runs the OPPONENT through the scripted `MacroStrategyExecutorOpponent` proxy, never the real sampled opponent, using `p.saveState`/`restoreState` to sandbox exploration away from the live match | `state: PyBulletMCTSState`, `strategy: MacroStrategy`, `horizon: int` | New `PyBulletMCTSState`; `float` (rollout value) | `pybullet` (indirectly, via `env`) | **Complete.** Proxy-vs-real drift never measured — diagnostic built this session, not yet run |
| `src/sbso/macro_executor.py` | `MacroStrategyExecutor`, `MacroStrategyExecutorOpponent`, `.to_pwm()`, `.__call__()` | Deterministic mapping from `MacroStrategy` to `(left_pwm, right_pwm)` — used BOTH as the agent's real-match action translator AND (wrapped as `MacroStrategyExecutorOpponent`) as MCTS's scripted opponent proxy | `strategy: MacroStrategy`, `tof`, `ir` | `tuple[float, float]` | `numpy` | **Complete** |
| `src/sbso/ablation_strategies.py` | `AblationConfig`, `.for_variant()` | Maps a config string (`none`/`no_sa`/`no_mcts`/`no_dspy`/`no_judge`) to 4 boolean component-enable flags | `strategy: str` | `AblationConfig(mcts_enabled, dspy_enabled, judge_enabled, sa_enabled)` | stdlib only | **Complete, validated against all 5 `phase4_*.yaml` variants this session** |
| `src/sbso/match_trainer.py` | `MatchLevelSBSOTrainer`, `._execute_strategy_live()`, `.run()` | THE canonical Phase 4 trainer — full matches, many decisions each, real MCTS search per decision, real live execution against the real sampled opponent | `ablation`, `mcts`, `opponent_pool`, `scheduler`, `checkpoint_mgr`, `dspy_compiler`, `env`, `executor`, `episodes: int`, `on_episode_end: Callable` | `dict` run summary; side effects: `self.training_pairs` (3-tuples `(episode,state,strategy)`), `self.recompile_history`, `self.mcts_calibration_log` | All of the above Phase 4 modules | **Complete, most heavily audited/fixed file this session** (episode-tagging, prompt-version tracking, calibration logging, state-replay validation gate — all added) |
| `src/sbso/training_loop.py` | `SBSOTrainer` | Older, Stage-1/2 one-decision-per-episode trainer (KNOWN SIMPLIFICATION per its own docstring); still used by `run_phase4_stage2_token_run.py` for fast plumbing proof | `root_state_builder: Callable`, `episodes: int` | `dict` run summary; `self.training_pairs` | Same Phase 4 modules, mock-friendly | **Complete, superseded by `MatchLevelSBSOTrainer` for real training but still the correct tool for cheap plumbing smoke tests** |
| `src/sbso/stage2_wiring.py` | `build_real_root_state_builder()`, `build_real_outcome_extractor()` | The two callables that let `SBSOTrainer` run against REAL physics/perception instead of Stage 1's mock placeholders, without changing `training_loop.py` itself | `backend: PyBulletSimulationBackend`, `perception_agent: PerceptionAgent` | `Callable` factories | `src.agents.perception_agent`, `src.sbso.simulation_backend` | **Complete** |
| `src/finetuning/lora_finetune.py` | `LoRAFineTuner`, `.run()`, `build_sft_dataset()`, `split_episodes_train_val()`, `evaluate_strategy_accuracy()` | SFT fine-tuning of Phi-4-mini via PEFT LoRA on accumulated training pairs; train/val split by episode; generation-based accuracy eval, incl. base-model comparison | `sft_records: list[dict]`, `eval_records: list[dict] \| None` | `dict[adapter_path, tokenizer, model, final_epoch_accuracy]` | `peft`, `transformers`, `datasets`, `torch` | **Complete, extensively fixed this session** (3-tuple unpack, `target_modules` for Phi-4-mini's fused QKV, train/val split, eval loop — all added/fixed) |
| `src/finetuning/merge_adapters.py` | `merge_lora_adapters()` | Merges LoRA adapters into the base model, fp16, HF format | `base_model_path`, `adapter_path`, `output_dir` | `Path` to merged fp16 model directory | `peft`, `transformers`, `torch` | **Complete** |
| `src/finetuning/quantize_gptq.py` | `quantize_gptq()` | INT4 quantization via GPTQ — **CUDA-only, cloud-only branch**, raises explicitly if no CUDA | `merged_model_path`, `calibration_texts: list[str]`, `bits: int=4` | `Path` to GPTQ output dir | `gptqmodel`, `transformers`, `torch` (CUDA build) | **Complete.** Cannot run on Mac by design |
| `src/finetuning/export_gguf.py` | `export_to_gguf()` | fp16 HF → GGUF-f16 conversion via `convert_hf_to_gguf.py` — **independent, parallel branch from GPTQ, not chained** | `merged_model_path` (NOT the GPTQ output — explicitly incompatible) | `Path` to GGUF-f16 file | Local `llama.cpp` checkout | **Complete, D4 fix applied** (previously incorrectly chained from GPTQ output) |
| `src/finetuning/quantize_gguf.py` | `quantize_gguf()`, `_find_llama_quantize_binary()` | Q4_K_M quantization via llama.cpp's own `llama-quantize` binary — no CUDA requirement, runs locally once llama.cpp is built | `gguf_f16_path`, `quant_type="Q4_K_M"` | `Path` to quantized GGUF | `llama-quantize` binary (PATH or explicit) | **Complete** |
| `src/finetuning/calibration_texts.py` | `extract_calibration_texts()`, `write_calibration_file()` | Pulls real, deduplicated, deterministically-sampled `lssd_text` strings straight out of `training_pairs` for GPTQ calibration — no separate generation run | `training_pairs: list`, `n_samples: int=256` | `list[str]`; `write_calibration_file()` → `Path` to newline-delimited text file | stdlib only | **Complete** |
| `src/finetuning/cost_projection.py` | `project_full_run()` | Pure arithmetic: pilot wall-clock → projected full-run cost/hours | `pilot_episodes`, `pilot_wall_clock_s`, `full_episodes_per_variant`, `num_variants`, `gpu_rate_usd_per_hr` | `dict[sec_per_episode, full_episodes_total, projected_hours, projected_cost_usd, gpu_rate_usd_per_hr]` | stdlib only | **Complete** |
| `src/data/dataset_registry.py` | (stub) | Tracks Phase 4 training vs. Phase 5a/5b holdout dataset separation | — | — | stdlib only | **STUB — Phase 0 scaffold, not implemented** |
| `scripts/run_phase4_pilot.py` | `main()`, `_make_opponent_factory()` | THE RunPod entrypoint — real SGLang, real PyBullet, real MCTS/Judge/DSPy, `MatchLevelSBSOTrainer` at pilot scale, cost projection | `config/phase4_pilot.yaml` (or `phase4_full_sbso.yaml`/ablation configs, same script) | `checkpoints/<run>/{progress.json, training_pairs.jsonl, prompt_history.jsonl, mcts_calibration.jsonl, run.log, gptq_calibration_texts.txt}` | Nearly every Phase 4 module | **Complete, most-fixed script this session** (wandb, structured logging, live cost/GPU telemetry, all logging wiring added) |
| `scripts/run_phase4_stage3_local.py` | `main()` | Same `MatchLevelSBSOTrainer`, wired for local Mac (LlamaCppJudge, MockDSPyCompiler default) — validates the SAME production trainer class, not a parallel implementation | `config/stage3_local.yaml` + CLI flags (`--episodes`, `--sim-budget`, etc. — intentionally CLI, not config, for local sweep speed) | Same artifact set as pilot | Same as pilot, minus SGLang | **Complete** |
| `scripts/run_phase4_stage2_token_run.py` | `main()` | Local, tiny-scale, incremental proof of every Stage 2 code path including a REAL LoRA fine-tune + merge; stops before GPTQ/GGUF (cloud-only) | CLI flags only — pure argparse, does NOT use `build_run()`/config system | LoRA adapters, merged fp16 model, `training_pairs.jsonl` | `src.finetuning.lora_finetune`, `src.finetuning.merge_adapters` | **Complete, LoRA split/eval wiring added this session** |
| `scripts/run_phase4_ablation.py` | `main()` | Stub entrypoint for a single ablation variant | `config/phase4_ablation_no_*.yaml` | — | `scripts._script_common` | **STUB — `build_run()` tuple bug fixed this session, phase logic still TODO** |
| `scripts/run_phase4_training.py` | `main()` | **Legacy, superseded** Stage-1-mock-only script (`MockJudge`/`MockDSPyCompiler`/`MockSimulationBackend`/`SBSOTrainer`) | `config` | — | Mock-only Phase 4 modules | **Superseded — not the real training path, do not confuse with `run_phase4_pilot.py`** |
| `scripts/run_export_pipeline.py` | `main()` | Orchestrates the merge→{GPTQ, GGUF-f16→quantize} two-branch export | `config/export_pipeline.yaml` | GPTQ dir + quantized GGUF file | `src.finetuning.{merge_adapters,quantize_gptq,export_gguf,quantize_gguf}` | Present, not specifically re-audited this session |
| `scripts/verify_a1_a2_real_pybullet.py` | `main()` | Real-vs-mock PyBullet parity check | — | — | `pybullet` | Present; platform-parity (Mac vs. RunPod) never reconfirmed on the actual RunPod box |
| `scripts/run_mcts_sim_budget_sweep.py` | `main()`, `_run_one_sweep_point()` | Runs the pilot trainer at multiple `sim_budget` values, compares win rate/strategy distribution/timing | `--sim-budgets`, `--episodes`, `--judge-model-path` | `results/mcts_sim_budget_sweep/sweep_summary.json` | `MatchLevelSBSOTrainer` + `LlamaCppJudge` | **Complete, built this session — local execution deprioritized** (LlamaCppJudge has no batching) |
| `scripts/backup_to_hub.py` | `backup_directory()` | Push LoRA/merged/GGUF artifacts to a private HF Hub repo — redundancy alongside a RunPod Network Volume | `--repo-id`, `--adapter-dir`, `--merged-dir`, `--gguf-dir` | Uploaded HF Hub repo contents | `huggingface_hub` | **Complete, built this session** |

## 3. Concrete Data Contracts & Interfaces

- **`training_pairs` shape**: `list[tuple[int, PyBulletMCTSState | dict, MacroStrategy]]` — episode-tagged 3-tuples. `build_sft_dataset()`, `calibration_texts.py`, and `RealDSPyCompiler._select_examples()` all duck-type the `state` element (`state.lssd_text` if object, `state["lssd"]` if dict).
- **`training_pairs.jsonl` persisted schema**: `{"episode": int, "lssd_text": str, "strategy": str}` per line.
- **`prompt_history.jsonl` schema**: `{"episode": int, "trigger_reason": "k_episodes"|"reward_drop", "prompt_version": int, "rolling_winrate": float|null, "accepted": bool|null, "validation": dict|null}`.
- **`mcts_calibration.jsonl` schema**: `{"episode": int, "decision_index": int, "chosen_strategy": str, "proxy_predicted_value": float, "match_outcome": str, "match_outcome_value": float}`.
- **`AblationConfig` variant strings**: `none` (full SBSO), `no_sa`, `no_mcts`, `no_dspy`, `no_judge` — exactly matching the 4 `phase4_ablation_no_*.yaml` filenames.
- **`config/phase4_*.yaml` required keys** (per `Phase4Config`): `mcts.{sim_budget, horizon, judge_prune_threshold}`, `dspy_recompilation.{k_rollout_batches, rolling_window_w, reward_drop_threshold_delta}`, `self_checkpoint_interval_episodes`, `episodes_total`. **Confirmed this session: `_shared_defaults.yaml` was missing the entire `mcts:` block** — fixed.
- **Export chain artifact types, explicitly non-interchangeable**: merged fp16 (HF format) → {GPTQ INT4 (HF-format, packed, SGLang-servable) | GGUF-f16 (llama.cpp-loadable, unquantized) → GGUF-Q4_K_M}. GPTQ output and GGUF are NOT chainable into each other — confirmed via `export_gguf.py`'s own D4-fix docstring.

## 4. Runtime Execution Lifecycle & State Management

1. **Per-decision cycle inside `MatchLevelSBSOTrainer`**: snapshot live state → `root_state = backend.root_state(lssd_text)` (implicitly a `p.saveState()`); `mcts.search(root_state)` runs `sim_budget` simulations, each sandboxed via further `saveState`/`restoreState` pairs inside `PyBulletSimulationBackend.step()`, with the opponent always played by the scripted `MacroStrategyExecutorOpponent` proxy during search; after search, `backend._restore(root_state.pybullet_state_id)` explicitly undoes all search branching, returning the live simulation to its pre-search state.
2. **Commit phase**: `_execute_strategy_live(best_strategy)` then drives the REAL live match forward for `decision_cycles` real physics steps, with `env.opponent_policy` restored to the REAL sampled opponent (rule-based/PPO/self-checkpoint) — the proxy substitution is strictly confined to the search sandbox.
3. **State ID lifecycle**: `_snapshot()` returns a PyBullet state ID; `_restore(id)` reloads it; `_free_ids()`/`release_search_states()` exist to bound memory growth from accumulated snapshot IDs across a long search — snapshot IDs not explicitly released persist in PyBullet's internal state table for the life of the physics client.
4. **DSPy recompile cycle**: `scheduler.should_recompile(ep, rolling_winrate)` checked every episode; on trigger, `dspy_compiler.compile(training_pairs, incumbent_prompt)` produces a candidate; if `validate_recompiles=True` (opt-in, not the pilot script's current default), `validate_prompt_candidate()` state-replays ~30 sampled states through both incumbent and candidate via a real `StrategyAgent` + `Judge.score_branch()`, accepting only if not worse within `accept_margin`.
5. **Self-checkpoint lifecycle**: `checkpoint_mgr.maybe_snapshot()` called every episode; stores `(episode, prompt_program, dominant_strategy)` in-memory only — `sample()` draws uniformly across the full accumulated list (confirmed matching report §3.2.1.3 exactly).
6. **Crash-safety, not crash-resume**: `training_pairs.jsonl`/`prompt_history.jsonl`/`mcts_calibration.jsonl`/`progress.json` are all written incrementally inside `on_episode_end`, so an interrupted run loses at most the current episode's data — but nothing in the trainer reconstructs `MatchLevelSBSOTrainer`'s in-memory state (opponent pool position, DSPy compiler state, checkpoint list) from these files; a restart begins a fresh run, not a resumed one.
7. **Concurrency**: within `MatchLevelSBSOTrainer` itself, execution is single-threaded/sequential — the only concurrency in Phase 4 is `SGLangJudge`/`SGLangSLMClient`'s internal `ThreadPoolExecutor` for batch scoring calls made FROM this sequential loop.

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **The single most severe finding of the entire audit, found and fixed this session**: `_shared_defaults.yaml` had no `mcts:` block at all, and `phase4_full_sbso.yaml` + all 4 ablation configs inherit only from it — **the real production run, as configured before this session's fix, would have crashed with `KeyError: 'mcts'` on the very first line of setup.** Confirmed fixed and validated; `phase4_pilot.yaml` was never affected (defines its own `mcts:` block directly).
- **`LlamaCppJudge`'s lack of batching is not merely slow, it is structurally incompatible with `judge_prune_threshold>0` at any meaningful `sim_budget`** — ~95 sequential 8B-model calls per decision at `sim_budget=15`. Confirmed empirically this session. Never use this backend for anything beyond a tiny local smoke test.
- **`sim_budget=20-40` and `c_uct=1.41` are uncalibrated** — not even flagged as placeholders in code (unlike `arena_config.yaml`'s honestly-labeled placeholders). Sensitivity sweep tooling exists; running it needs `SGLangJudge` (RunPod), not the local `LlamaCppJudge` path.
- **MCTS never validates its scripted-proxy assumption against reality** — the opponent during search is always the same generic script regardless of whether the real sampled opponent is rule-based, PPO, or a self-checkpoint SLM. Diagnostic exists (`report_mcts_judge_calibration.py`), unrun.
- **The Judge's scores have never been checked against real match outcomes** — same diagnostic covers this; genuinely unknown whether Judge-based pruning is discarding good branches or keeping bad ones.
- **VRAM contention risk (Phase 3 cross-reference)**: running GPTQ quantization while both SGLang servers are still up risks exceeding available GPU memory (`0.45 + 0.45 = 0.90` already committed).
- **`gpu_rate_usd_per_hr` placeholders**: `phase4_pilot.yaml` ships `1` (a round-number placeholder, flagged); `cost_projection.py`'s own code default (`0.34`) matches current real RunPod RTX 4090 Community pricing far more closely — worth using the code default or an explicitly researched value, not the YAML placeholder, for any real projection.
- **No crash-RESUME, only crash-SAFE incremental logging** — a RunPod interruption loses in-memory trainer state (opponent pool position, DSPy/checkpoint state) even though the JSONL logs survive. Mitigation: on-demand (non-spot) instance + `tmux` + (optionally) `backup_to_hub.py`/a Network Volume for the artifacts, not a code-level resume capability, which does not exist.
- **`run_phase4_training.py` is legacy/superseded** — still present in the repo, still importable, still uses only Mock components. A risk of confusion (someone runs this instead of `run_phase4_pilot.py` expecting real training) rather than a functional bug.

---

# CROSS-CUTTING INFRASTRUCTURE & LOGGING

## 1. Architectural Role & Objective

Infrastructure that no single phase owns but every phase depends on: config
validation (Phase 0, applies everywhere), structured logging, experiment
tracking, live cost/GPU telemetry, deployment scripts, and diagnostic tooling
built specifically to answer "is this pipeline producing genuine results"
rather than "does this code run."

## 2. Comprehensive File & Script Registry

| File Path | Core Classes / Functions | Operational Responsibility | Input Format & Data Types | Output Format & Data Types | Key Dependencies | Current Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `scripts/_script_common.py` | `setup_logging()`, `poll_gpu_stats()` | Dual stdout+file logging (`checkpoints/<run>/run.log`); defensive `nvidia-smi` polling, returns `None` (never raises) on non-GPU machines | `out_dir: Path` | `logging.Logger`; `dict[gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb] \| None` | stdlib `logging`, `shutil`, `subprocess` | **Complete, built this session.** GPU-poller confirmed safe on non-GPU machines; real values need RunPod confirmation |
| `scripts/preflight_static_check.py` | 6 check functions, `CheckSuite` | Local, no-GPU config/wiring checks: DSPy trigger nulls, `build_run()` tuple consistency (catches the class of bug that recurred 5x this session), malformed-YAML-value detection, HCMA-not-in-Phase-4, GPU-rate placeholder detection | Repo root path | Console report + exit code | stdlib only | **Complete, tested against real repo state** |
| `scripts/preflight_runpod_check.py` | 4 check functions | RunPod-only runtime checks: GPU visible, SGLang servers reachable, constrained decoding schema-valid under real load, `pytest` on the actual cloud image | Live RunPod environment | Console report + exit code | `requests`, `nvidia-smi`, `pytest` | **Complete.** Cannot be executed/verified outside a real RunPod session |
| `scripts/run_mirror_match.py` | `main()`, `_load_pair()` | Symmetric-baseline sanity check: identical controller vs. itself, checks win rate ≈ 50% among decided matches | `--attacker`, `--episodes`, `--randomize` | `results/mirror_match/results.jsonl`; console win/loss/draw summary | `src.baselines.*`, `src.simulation.sumo_env` | **Complete, run for real this session** — 74W/69L/57D, 51.7%/48.3% among decided, no structural asymmetry confirmed |
| `scripts/run_stress_test.py` | `main()`, `_load_attacker_policy()` | Out-of-family generalization test: named baseline vs. `BaitController`; `--gui`/`--trace`/`--episode-seconds`/`--randomize-rule-based` | Same as above + episode-length override | `results/stress_test_bait/results.jsonl` | Same + `src.baselines.adversarial_bait_controller` | **Complete, iterated to a legitimate final result this session** (§2 of session record) |
| `src/evaluation/strategy_distribution.py` + `scripts/report_strategy_distribution.py` | `compute_distribution()`, `compute_distribution_by_prompt_version()` | Macro-strategy usage % from `training_pairs.jsonl`; optional breakdown by DSPy prompt version (joins with `prompt_history.jsonl` by episode range) | `training_pairs.jsonl` path, optional `prompt_history.jsonl` path | Console report; optional JSON dump | stdlib only | **Complete, tested end-to-end against synthetic + real-shaped data** |
| `scripts/report_mcts_judge_calibration.py` | `compute_calibration()`, Pearson correlation | Correlates MCTS search-time predicted value against real match outcome — one diagnostic answering both "is the proxy opponent representative" and "is the Judge calibrated" | `mcts_calibration.jsonl` path | Console report incl. binned predicted-vs-actual table | stdlib only (`math`) | **Complete, tested against synthetic data with a known relationship** |
| `scripts/validate_quantization_quality.py` | `run_comparison()`, `_GGUFModel` | Compares fp16-vs-quantized GGUF decision agreement + schema-validity on sampled validation prompts | `--fp16-gguf-path`, `--quantized-gguf-path`, `--validation-prompts` | Console report; agreement rate, per-model schema-valid rate | `llama-cpp-python` | **Complete, cannot run until real export artifacts exist** |
| `scripts/generate_report_tables.py` | 4 table functions, `format_markdown_report()` | Regenerates sensitivity/robustness thesis tables: sim_budget sweep, stress-test legitimacy, mirror-match balance, DSPy recompile summary | Paths to the 4 source JSONL/JSON files above | `results/report_tables/REPORT.md` | `scripts._script_common` | **Complete, tested against synthetic data reproducing this session's real numbers exactly** |
| `scripts/backup_to_hub.py` | `backup_directory()` | Off-RunPod artifact redundancy (see Phase 4 registry) | — | — | `huggingface_hub` | **Complete** |
| `scripts/launch_sglang_servers.sh` | `_start_one()` | Launches both SGLang servers in detached `tmux` sessions | `agent_model_path` (optional), `judge_model_path` (required), ports, `mem_fraction` | Two running `tmux` sessions (`sglang-agent`, `sglang-judge`) | `tmux`, `sglang` (server-side) | **Complete, stale filename references fixed this session** (referenced a deleted script) |
| `requirements.txt` | — | Base/local dependencies | — | — | — | `wandb`, `huggingface_hub` added this session. **`requirements-cloud.txt` exists in the real repo (confirmed via handover doc) but was not available for this audit — verify it independently covers `transformers`/`peft`/`datasets`/`sglang` and this session's additions before the run** |
| `wandb` integration (cross-file) | — | PPO: `sync_tensorboard=True` (reuses existing `tensorboard_log`); LoRA: `report_to=["wandb"]`; SBSO loop: `wandb.log()` in episode-end hook incl. live `cost_so_far_usd` and GPU stats | `--use-wandb` flag on 4 scripts | Live W&B dashboard | `wandb>=0.16` | **Complete, wiring verified; requires `wandb login` before first use** |

## 3. Concrete Data Contracts & Interfaces

- **`run.log` format**: `%(asctime)s [%(levelname)s] %(message)s`, `HH:MM:SS` timestamp — both console and file get identical formatting.
- **`results.jsonl` (mirror-match)**: `{"episode": int, "attacker": str, "outcome": str, "randomized": bool}`.
- **`results.jsonl` (stress-test)**: `{"episode": int, "attacker": str, "outcome": str, "reason": str|null, "min_dist": float, "randomized": bool, "episode_seconds": float|null}`.
- **`sweep_summary.json`**: `list[dict[sim_budget, episodes, elapsed_seconds, win_rate, judge_call_count, avg_decision_seconds, training_pairs_path]]`.
- **GPU poll dict**: `{"gpu_utilization_pct": float, "gpu_memory_used_mb": float, "gpu_memory_total_mb": float}`.

## 4. Runtime Execution Lifecycle & State Management

- **Logging**: `setup_logging()` called once per script's `main()`, immediately after `out_dir` is known; `logger.handlers.clear()` guards against duplicate handlers if a process ever calls this twice.
- **wandb**: `wandb.init()` called once per run if `--use-wandb`; `wandb.log()` called from inside each script's `on_episode_end` hook (same cadence as JSONL persistence); `wandb.finish()` called once at run end.
- **GPU polling**: called synchronously inline within the episode-end hook, adding one subprocess call (`nvidia-smi`, ~10ms typical) per logged episode — not a background thread.
- **Preflight checks**: intended as manual, deliberate gates — run once before provisioning (`preflight_static_check.py`) and once after servers are up (`preflight_runpod_check.py`), not integrated into the training scripts themselves.

## 5. Critical Remarks, Edge Cases & Deployment Risks

- **`requirements-cloud.txt` could not be audited** — genuinely exists per the handover doc, not available in this session's file access. This is the single largest unverified risk in this entire report: if it's missing `transformers`/`peft`/`datasets`, the LoRA path crashes with `ImportError` on RunPod; if missing `wandb`/`huggingface_hub`, this session's tooling does too. **Verify directly before the run.**
- **`wandb` requires outbound internet from the RunPod pod** and a valid API key (`wandb login`) — if the pod has restricted egress, `wandb.init()` will hang or fail; no fallback/offline-mode handling has been added.
- **GPU polling silently returns `None` everywhere except a real GPU machine** — correct, defensive behavior, but means this entire telemetry path is **completely unverified in practice** until first run on real RunPod hardware.
- **The diagnostic scripts (`report_mcts_judge_calibration.py`, `validate_quantization_quality.py`, `run_mcts_sim_budget_sweep.py`) are all built and unit-tested against synthetic data, but NONE have been run against real Phase 4 output yet** — their correctness on real data remains unconfirmed until the pilot run produces real `mcts_calibration.jsonl`/GGUF artifacts.

---

# MASTER CROSS-PHASE DEPENDENCY & ARTIFACT FLOWCHART

```
PHASE 0 (config/run infra)
  config_loader + config_schemas + run_context
        │  (RunContext, validated config)
        ▼
PHASE 1 (physics + baselines)
  arena.py + robot.py + sensors.py ──► PyBulletSumoEnv
        │ tof[7], ir[2], encoder[4]         │
        │ (raw np.ndarray)                  │ RuleBasedController / PPOController
        ▼                                   │ (Baseline 1 / Baseline 2, no SLM)
PHASE 2 (perception + multi-agent reasoning)
  SavitzkyGolayFilter + IRGradientFilter + MotorDeadbandFilter
        │
        ▼
  LSSDEncoder ──► "opp=near,dir=FC;edge=safe;mom=fwd"  (LSSD text)
        │
        ▼
  PerceptionAgent ──► PerceptionState (pydantic)
        │
        ├──► OpponentAnalysisAgent ──► OpponentAnalysis (t-1, pipelined)
        │
        ▼
  StrategyAgent ──► MacroStrategyDecision {charge|flank|retreat|hold|evade_edge}
        │
        ▼
  TacticalExecutionAgent ──► TacticalCommand {7 TacticalKeyword values}
        │
        ▼
  ActuatorBridge ──► (left_pwm, right_pwm) ∈ [-1,1]²
        │
        ▼  [all above orchestrated by LangGraph StateGraph / NODE_SEQUENCE,
        │   agents depend only on the abstract SLMClient interface]
PHASE 3 (real inference backends, swapped in without touching Phase 2 code)
  MockSLMClient (Phase 2 dev)
        ⇩ swap
  LlamaCppSLMClient (local, GBNF-constrained, no batching)
  SGLangSLMClient   (cloud, regex-constrained, concurrent ThreadPoolExecutor)
        │
        │  HELProfile / HELLogger (latency emulation - INACTIVE except Phase 5c)
        ▼
PHASE 4 (SBSO training loop, per real decision inside a live match)
  root_state = backend.root_state(lssd_text)        [p.saveState() snapshot]
        │
        ▼
  MCTS.search(root_state)  ×sim_budget simulations
        │  opponent during search = MacroStrategyExecutorOpponent (SCRIPTED PROXY,
        │  never the real sampled opponent - by design, for speed)
        │  Judge.score_branch() / score_position() prunes/evaluates nodes
        ▼
  best_strategy  ──►  backend._restore(root_state)  [undo search branching]
        │
        ▼
  _execute_strategy_live(best_strategy)  [REAL opponent restored, REAL physics
        │                                 advances decision_cycles steps]
        ▼
  training_pairs.append((episode, state, best_strategy))
        │
        │◄── RecompilationScheduler.should_recompile() ──► RealDSPyCompiler.compile()
        │        (K episodes OR reward-drop trigger)         [+ validate_prompt_candidate()
        │                                                      state-replay gate, opt-in]
        │◄── SelfCheckpointManager.maybe_snapshot() (every 500 episodes → 10 checkpoints)
        │◄── OpponentPool.sample() next episode's opponent (baseline1/baseline2/self_checkpoint)
        ▼
  [loop until episodes_total (5000) reached, or ablation-scoped variant]
        │
        ▼
  training_pairs.jsonl  (full episode-tagged corpus, one of 5 runs: Benchmark2 + 4 ablations)
        │
        ▼
  LoRAFineTuner.run()  [split_episodes_train_val() → SFT train/val split]
        │
        ├──► LoRA adapters (PEFT format)
        │
        ▼
  merge_lora_adapters()  ──►  merged fp16 model (HF format)
        │
        ├─────────────────────────────┬─────────────────────────────────
        ▼ (branch A: cloud serving)    ▼ (branch B: edge/HEL deployment)
  quantize_gptq()                export_to_gguf()  [fp16 → GGUF-f16]
  [CUDA-only, RunPod]                   │
        │                               ▼
        ▼                        quantize_gguf()  [llama-quantize]
  GPTQ INT4 model                       │
  (SGLang-servable)                     ▼
                                  Q4_K_M GGUF binary
                                  (llama.cpp-servable, HEL/Jetson target)
        │                               │
        └───────────────┬───────────────┘
                         ▼
          [OUT OF SCOPE: Phase 5 evaluation - match_runner.py,
           win_rate.py, elo.py, statistical methodology - all
           still Phase 0 stubs, not yet implemented]
```

**Two artifact families reach a terminal state at the bottom of this diagram**,
and they are deliberately non-interchangeable: the GPTQ branch serves via
SGLang for Phase 5a/5b full-power cloud evaluation; the GGUF/Q4_K_M branch
serves via llama.cpp for Phase 5c's HEL-throttled edge-deployment evaluation.
Both originate from the identical merged fp16 checkpoint, produced exactly
once per training run, per the report's own §3.2.1.3-adjacent design.
