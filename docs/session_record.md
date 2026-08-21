# Sumo-SBSO — Session Record

Full thread, Phase 4 pre-flight audit through pre-deployment sign-off. Chronological
by topic. Supersedes the earlier version of this file.

---

## 1. Bugs found and fixed (chronological)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `lora_finetune.py` | `build_sft_dataset()` unpacked `training_pairs` as a 2-tuple; real data is episode-tagged 3-tuples | Unpack 3-tuple |
| 2 | `lora_finetune.py` | Default `target_modules` used Llama-style names; Phi-4-mini fuses Q/K/V into `qkv_proj` — PEFT silently trained `o_proj` only | `["qkv_proj","o_proj","gate_up_proj","down_proj"]` |
| 3 | `finetuning.yaml` | `lora:` block never actually read | Wired into `run_phase4_stage2_token_run.py` |
| 4 | `scripts/run_phase4_ablation.py` (+4 more) | Stale `build_run()` 2-tuple unpack | All 5 fixed (`run_phase4_ablation.py`, `generate_report_tables.py`, `run_phase5{a,b,c}_eval.py`) |
| 5 | `baseline2_ppo.yaml` | `shaping_push_weight: float = 0.05` — Python syntax pasted into YAML, parses as a string | Harmless (never read) — flagged, not fixed |
| 6 | `robot.py` | Wheel-turn-direction sign inverted — confirmed empirically (-124° when +expected) | Swapped `left_wheel_joints`/`right_wheel_joints`. **User-confirmed: all tests pass.** |
| 7 | `arena.py` / `sumo_env.py` | No capsize detection — a tipped robot triggered neither `is_outside()` nor `has_fallen()`, ran to timeout as an uninformative draw | Added `has_capsized()` (roll/pitch), wired into terminal check |
| 8 | `sumo_env.py` | No record of *why* a match ended | Added `agent_out_reason`/`opponent_out_reason` to `info` |
| 9 | `ir_gradient.py` | Returned a raw window delta, not a true per-second rate | Normalized by `dt_s`, call sites updated |
| 10 | `_shared_defaults.yaml` | `k_rollout_batches=5`/`rolling_window_w=10` vs. scheduler's documented `K=500`/`W=100` (report §3.2.1.3 confirms K=500 by design) | **User confirmed: `k_rollout_batches` set to 500.** `W`/`delta` still open |
| 11 | `match_trainer.py` | DSPy recompile trigger reason computed then discarded; no prompt-version history | Added `prompt_version`/`recompile_history`, persisted to `prompt_history.jsonl` |
| 12 | `_shared_defaults.yaml` | **`mcts:` block missing entirely** — `phase4_full_sbso.yaml` + all 4 ablation configs would crash with `KeyError: 'mcts'` on startup, found by the new config validator's first real test | Added `mcts:` block (`sim_budget: 40`, explicit placeholder) |
| 13 | `config_schemas.py` | `PHASE_SCHEMAS` registered `"phase4"` but `run_phase4_pilot.py` calls `build_run(phase="phase4_pilot")` — validation silently never covered the actual pilot script | Registered `"phase4_pilot"` too. Deliberately NOT `"phase4_stage3_local"` — that config is intentionally CLI-driven with no `mcts:` block; validating it would introduce a new crash |
| 14 | `launch_sglang_servers.sh` | Comments reference `run_phase4_stage3_cloud.py`, confirmed deleted this session (per handover doc) and replaced by `run_phase4_pilot.py` | Both references corrected |

## 2. `adversarial_bait_controller.py` (`BaitController`) — full iteration arc

Built as an out-of-family stress-test opponent (lure → counter) to test whether
trained/baseline models generalize past chargers.

1. **Stalemate loop bug**: `counter_timer` re-armed forever once locked in contact — all-identical draws (fully deterministic setup, zero variation). Fixed: consecutive-retrigger counter forces a real disengage.
2. **Retreat-toward-own-edge bug**: straight local-frame reverse, no boundary awareness. Fixed: soft IR-based edge-repulsion blended into the retreat steer.
3. **Never won legitimately (500 episodes, first real randomized run)**: 0/14 losses were `pushed_out`, all 14 `capsized`. Root cause: `counter_max_consecutive_triggers=3` broke off genuinely-winning pushes before completion (`rule_based`'s `commit_cycles=10` sustains pressure uninterrupted). Fixed: raised to 10.
4. **Final result (500 episodes, post-fix)**: win-pushed_out 347, win-capsized 69, loss-pushed_out 23, loss-capsized 6. **First legitimate tactical wins (23/500, up from 0)**. 83.1% of decided outcomes are genuine pushes. `rule_based` still wins ~93.5% of decided matches — a legitimate finding (blunt commitment beats cautious counter-play most of the time, but not always), not a bug.

## 3. Mirror-match / stress-test results (as reported)

- Wheel-direction fix: all `test_wheel_turn_direction.py` tests pass.
- Mirror match (`rule_based` vs. itself, 200 episodes): 74W/69L/57D → 51.7%/48.3% among decided (well within statistical noise, n=143) — **no structural asymmetry**. Draw rate 28.5%.
- Stress test final (500 episodes): see §2 item 4.

## 4. Critical config finding — full production run could not have started

Building the config-validation framework (Part 2 discussion, item 2) surfaced a
severe, previously invisible bug on its very first real test: `phase4_full_sbso.yaml`
and all 4 ablation configs only ever `extends: _shared_defaults.yaml`, which had **no
`mcts:` block at all** — while `run_phase4_pilot.py` unconditionally reads
`config["mcts"][...]` at startup. **All 5 real production configs would have crashed
immediately with `KeyError: 'mcts'`.** Fixed (§1 item 12). A second, related gap
(§1 item 13) meant even after that fix, validation wasn't actually running against
the real pilot script until caught in the final pre-deployment audit pass.

## 5. New tooling built this session

| File | Purpose |
|---|---|
| `src/evaluation/strategy_distribution.py` | Macro-strategy usage distribution; breaks down by DSPy prompt version via `prompt_history.jsonl` |
| `scripts/report_strategy_distribution.py` | CLI wrapper |
| `src/baselines/adversarial_bait_controller.py` | Out-of-family stress opponent (§2) |
| `scripts/run_stress_test.py` | Harness — `--gui`, `--trace`, `--episode-seconds`, `--randomize-rule-based`; now writes `results.jsonl` |
| `scripts/run_mirror_match.py` | Symmetric-baseline sanity check; now writes `results.jsonl` |
| `scripts/preflight_static_check.py` | Local, no-GPU config/wiring checks |
| `scripts/preflight_runpod_check.py` | RunPod-only runtime checks (GPU, SGLang servers, constrained decoding under load, pytest) |
| `tests/unit/test_wheel_turn_direction.py` | Wheel-direction regression test — **passing** |
| `scripts/validate_quantization_quality.py` | fp16-vs-quantized GGUF agreement comparison — built, not yet runnable (needs real export artifacts) |
| `scripts/run_mcts_sim_budget_sweep.py` | `sim_budget` sensitivity sweep — built; **local run deprioritized** (LlamaCppJudge has no batching, ~95 sequential judge calls/decision made even 10 episodes impractically slow; revisit on RunPod with SGLangJudge instead) |
| `src/common/config_schemas.py` | Pydantic per-phase config validation (`Phase4Config`, `Phase1Config`) |
| `src/sbso/dspy_compiler.py::validate_prompt_candidate()` | State-replay DSPy prompt validation (no new episodes) — design recorded in memory for handover |
| `src/finetuning/lora_finetune.py::split_episodes_train_val()` / `evaluate_strategy_accuracy()` | LoRA train/val split (random by episode, zero leakage verified) + generation-based accuracy eval, incl. base-model-vs-LoRA comparison |
| `scripts/report_mcts_judge_calibration.py` | Correlates MCTS search-time predicted value vs. real match outcome — answers both the "is the proxy opponent representative" and "is the Judge calibrated" questions with one shared log |
| `scripts/backup_to_hub.py` | Push LoRA/merged/GGUF artifacts to a private HF Hub repo (item 5, Layer 2) |
| `scripts/generate_report_tables.py` | Sensitivity/robustness tables (sim_budget, stress-test legitimacy, mirror-match balance, DSPy recompile summary) — tested against synthetic data reproducing this session's real numbers exactly |
| `wandb` integration | PPO (`sync_tensorboard=True`, reuses existing `tensorboard_log`), LoRA (`report_to=["wandb"]`), SBSO loop (`wandb.log()` in episode-end hook incl. live cost accumulation + GPU stats) |
| Structured logging | `setup_logging()` in `_script_common.py`; all 4 training scripts fully converted from `print()` |
| `_script_common.py::poll_gpu_stats()` | Defensive `nvidia-smi` polling, confirmed safe on non-GPU machines |

## 6. Open items, prioritized

1. **`W`/`delta` for DSPy recompilation** — still needs pilot calibration; `K` is settled.
2. **`max_tilt_rad=1.0` capsize threshold** — reasonable starting point, untuned.
3. **Rule-based capsize-fragility root cause** — 6/500 residual capsize losses even post-fix; not yet investigated (see current discussion below).
4. **`baseline2_ppo.yaml`'s malformed shaping-weight syntax** — harmless, cleanup only.
5. **`sim_budget`/`c_uct`** — never calibrated; sweep script exists, deprioritized locally.
6. **Multiple-comparisons correction for Phase 5 statistics** — explicit KIV.
7. **`requirements-cloud.txt`** — referenced by 2 files, confirmed to genuinely exist in the real repo, but not available for audit here. **User should verify it covers `transformers`/`peft`/`datasets` and this session's `wandb`/`huggingface_hub` additions before the run.**
8. **End-to-end integration test (train→export→quantize→eval)** — blocked on Phase 5's match-runner, which doesn't exist yet.

## 7. Key methodological findings for the thesis

- **Opponent lineage was a single behavioral archetype at its root** — now partially addressed: `BaitController`'s 23/500 legitimate wins are the first evidence the training/baseline lineage can be beaten by genuinely different tactics, not just narrow counter-tuning.
- **MCTS searches a scripted proxy opponent during tree search, never the real sampled opponent** — by design (self-checkpoint opponents would need live SLM inference inside the search loop otherwise). Diagnostic built (`report_mcts_judge_calibration.py`), not yet run against real data.
- **The Judge's scores have never been checked against real match outcomes** — same diagnostic covers this.
- **No crash-resume for RunPod** — mitigated via on-demand instance + `tmux` discipline + existing incremental saves; a Network Volume (infra choice, not code) is the recommended durability layer, with `backup_to_hub.py` as secondary redundancy for final artifacts.
- **PPO/rule-based diagnostic**: `rule_based`'s blunt commit-charge (`commit_cycles=10`, uninterrupted) is structurally advantaged in any push contest and occasionally self-capsizes — directly relevant to whether PPO's training signal (trained against `rule_based`) is clean. See current discussion.

---

## 8. Current discussion: does `rule_based_controller.py` need enhancement, and does it affect PPO?

Raised because `rule_based` still capsizes 6/500 times (1.2%) even after both `BaitController` fixes, and `rule_based` is literally what PPO trains against. Two separable questions:

- **Physical stability** (capsize root cause) vs. **decision-logic sophistication** (blunt vs. tactical) are different fixes with different justifications — worth not conflating.
- **Timing matters most**: if `rule_based` is going to change at all, it must happen **before** PPO trains — changing it after would invalidate/require redoing PPO training, since PPO's policy is tuned against `rule_based`'s current specific behavior.
- **Evidence bar not yet met for a decision-logic change**: no direct evidence PPO or Benchmark 2 specifically *exploit* `rule_based`'s bluntness (vs. just beating it fairly); the report frames Baseline 1 as deliberately simple by design.
- **Evidence for physical-stability investigation is stronger but still thin**: 6/500 is a low rate, single opponent (`BaitController`), no data yet on whether it's noise or a genuine, exploitable fragility (e.g. tied to specific extreme randomized param draws).
- **Recommended next step, not yet executed**: a larger stress-test sample and/or checking whether capsizes correlate with specific `RuleBasedParams` draws (e.g. max `charge_speed`), before deciding whether this needs a fix — cheap to check, avoids both under- and over-reacting.
