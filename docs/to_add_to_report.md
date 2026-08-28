The ScriptedSLMClient is more than a test helper — it's how you'll build deterministic end-to-end integration tests and, later, how you can inject a known strategy to isolate whether a failure is in reasoning vs. execution. It's the mock equivalent of forcing a specific macro-intent.

The staleness stamping (frame_stamp on OAA, read by SA with an explicit "classification from frame N" note in the prompt) makes the pipelined design honest and debuggable: when you later analyze why SA chose a strategy, you can see exactly how old the opponent read was. That's a real methodological point — the report can state that SA operates on a bounded-staleness (≤1 frame ≈ 50ms) opponent model, which is the latency-hiding mechanism that lets the pipeline approach the decision window.

The single-source-of-truth node design (pure functions, run by either a plain loop or LangGraph) means your thesis can honestly say the system uses LangGraph orchestration, while your dev/CI stays fast and model-free. The LangGraph graph and the pure runner execute identical logic — one implementation, two executors.

The timing dict now captures per-node latency every cycle. Right now with the mock those numbers are microseconds (instant), but this is exactly the instrumentation Phase 5c needs to measure real decision latency against the 50ms window. When Phase 3 swaps in real Phi-4-mini, these numbers become the actual latency data — and they'll show the honest gap we discussed.

HCMA is built but not yet wired into the cycle. I deliberately left it as a standalone policy object rather than forcing it into the decision loop, because it needs the real HEL latency feed (Phase 3) to be meaningful — feeding it a mock "always full headroom" now would just be theater. In Phase 3, the controller will consult hcma.compute_token_budget(...) before each SLM call and hcma.should_bypass_sa(...) after OAA, using real HEL timing. The policy logic is tested and ready; it just needs a real latency signal to act on.

### Issue 3
On "will benchmarks dominate the game" — plausibly yes, and if the diagnostics above show low strategy diversity, the honest framing for the writeup is "competitive/dominant within this study's opponent distribution," not "generally tactically superior" — which is a defensible, gradable claim on its own, and arguably closer to what a WQD7025-scope project can actually support given the single-lineage opponent pool. I'd treat this as a Limitations-section item to write up explicitly (Section 3.6 is exactly where it belongs) rather than something that has to be fully solved before submission — a fully independent adversary population is a much bigger lift than your timeline probably has room for.Quick correction first: rule_based_controller.py isn't a fixed predefined-direction walker — it's a reactive priority stack (edge-avoid → attack-toward-detected-opponent → search-and-creep), and run_phase1_baselines.py already trains PPO against RuleBasedParams.randomized() — a fresh randomized instance per episode, not one frozen instance (that fix is already in, per the Phase 1 audit). So "PPO memorized one exact rule-based pattern" specifically isn't what's happening.

But there's a deeper version of your concern that is real, and the report's own methodology section (3.2.1.2) walks right up to it without fully resolving it. The report's argument is: mixing Baseline 1 (rule-based) + Baseline 2 (PPO) + self-checkpoints in the training opponent pool prevents narrow memorization and "proves generalization, not memorized counter-play." The problem: those three aren't as independent as that framing suggests. PPO is trained against rule-based. Self-checkpoints are Benchmark 2's own past selves, which were themselves trained against rule-based and PPO. Trace the lineage back far enough and every opponent Benchmark 2 ever faces is a descendant of one original behavioral archetype — the reactive rule-based state machine. Parameter randomization gives PPO diversity in thresholds and speeds, not diversity in strategic behavior classes (no baiting, no feinting, no deliberate edge-luring anywhere in the lineage). That's a real risk of what's usually called self-play/co-evolutionary monoculture: everyone in the tower gets very good at beating variations of the same underlying idea, and a high win rate across Tier 1/2/3 would then reflect dominance within that closed family, not general tactical superiority — exactly the distinction your Q1 conversation was already circling.

How to check this, cheaply, without building new opponents first:

Macro-strategy usage distribution. You already have the 5 MacroStrategy values (charge, flank, retreat, hold, evade_edge) logged as part of every training pair. Just count and report the distribution for Benchmark 2. If it converges to ~all charge, that's direct evidence of a narrow-opponent artifact — a reactive, non-baiting opponent family never punishes pure aggression, so the SLM never needed flank/hold/evade_edge to win. This is nearly free to compute and far more diagnostic than win rate alone.
Self-checkpoint dominant_strategy diversity. self_checkpoint_manager.py already tracks this field per checkpoint. Check whether it actually varies across the 10 accumulated checkpoints, or collapses to one dominant strategy early and stays there — that's the self-play-collapse signature directly, not inferred.
One out-of-family stress test, outside the official five conditions. Not a new RQ, not a KPI — just one deliberately different hand-authored opponent persona (e.g., a baiting/retreat-then-counter controller, or RuleBasedParams pushed to an extreme evasive profile you don't already sample from) run against the final Benchmark 2 checkpoint. If win rate holds up there too, that's real evidence of generality. If it collapses relative to the official-condition numbers, that's the clearest possible proof the earlier dominance was a monoculture artifact, and it costs you one extra opponent controller, not a new training run.

# Issue 4 to check
Bugs found and fixed this session: build_sft_dataset crashed on real (episode-tagged) training data — fixed. LoRA target_modules silently trained only o_proj on Phi-4-mini's fused-QKV architecture instead of the full attention block — fixed, both in code and finetuning.yaml. finetuning.yaml's LoRA hyperparameters were dead config, never actually read — wired through. Five scripts had a stale build_run() 2-tuple unpack that crashes on use — one fixed, four flagged. baseline2_ppo.yaml had three shaping-weight values written as invalid Python syntax instead of YAML, parsed as strings — currently harmless only because they're never read, flagged for cleanup.

Methodology risks surfaced, not yet resolved: DSPy recompilation frequency (K=5, W=10) diverges 10-100x from the scheduler's own documented intended defaults (K=500, W=100) — needs a deliberate pilot-calibrated decision, not a default carried forward unexamined. The training opponent lineage (rule-based → PPO → self-checkpoints) is one behavioral archetype at its root — confirmed empirically via the mirror-match observation (both baselines rush straight in, every episode) — meaning early win-rate dominance should be reported as "within this study's opponent distribution," not general tactical superiority, unless the out-of-family stress test says otherwise. MCTS searches a scripted proxy opponent, never validated against real opponent behavior. The LLM-as-Judge's scores are never checked against real match outcomes. No crash-resume on RunPod, no checkpoint backup off the ephemeral pod, no experiment tracking beyond stdout/JSON.

Tooling built this session, ready for later use: macro-strategy usage distribution counter, an out-of-family adversarial stress-test opponent (BaitController) plus a CLI harness to run it, and a two-part preflight framework (local static checks + RunPod-only runtime checks) that automatically catches the build_run()-tuple and malformed-YAML bug classes going forward.

Open, unresolved as of this conversation: wheel-turn-direction sign (test written, not yet run to a real verdict), ir_gradient.py's missing time-normalization, BaitController's standoff-deadlock behavior (diagnosed from your trace, fixed, not yet re-verified against a live run).

# print logging (error when runpod crash)
4. print() → logging — what actually changes for you

Nothing about how you invoke any script. Same commands, same flags — this is purely internal to how output gets written, not what you type.

What you get that you don't have today: right now, once a tmux session's scrollback fills up or you close the terminal, that output is gone — the only record of a multi-hour run is whatever's still in scrollback. After this change, every run also gets a permanent file, checkpoints/<run_id>/run.log, with a timestamp and severity level on every line, that survives closing the terminal entirely.

Concretely, what this is for:

Post-mortem after a crash or a long unattended run — you can open run.log the next morning and see the whole history, not just whatever happened to still be in your terminal.
grep ERROR run.log or grep WARNING run.log — instantly surfaces problems (a rejected DSPy candidate, a schema fallback firing) without scrolling through thousands of lines of normal progress.
Timestamps — lets you see exactly how long each phase actually took after the fact, feeding directly into the cost/timing tracking from item 1.
Cleaner future integration — structured log records are easier to feed into wandb/a dashboard later than raw print() strings, if you go that route.

## MCTS 
# Self-Checkpoint Opponent: Fix Applied + Known Limitation for Report

## 1. Fix applied — `scripts/run_phase4_pilot.py` (KeyError blocker)

**Symptom:** running `run_phase4_pilot.py` against `phase4_full_sbso.yaml` or any
ablation config crashed immediately with `KeyError: 'pilot_scope'`.

**Root cause:** the opponent factory read `config["opponent_pool"]["pilot_scope"]`
directly, a key that only exists in `phase4_pilot.yaml`. Separately, the
`OpponentPool(...)` construction never passed `self_checkpoint_manager` or
`target_counts`, so even without the crash, self-checkpoint opponents (the model
playing against snapshots of its own past strategies) could never be sampled —
`OpponentPool.sample()` requires `self_checkpoint_manager is not None` before it will
ever offer that choice.

**Fix:**
- `_make_opponent_factory` is now called with a hardcoded `["baseline1", "baseline2"]`
  instead of a config read. This scope is invariant between pilot and full run — the
  factory only ever resolves baseline opponent kinds into policies; self-checkpoint
  opponents carry their own `rollout_policy` already embedded on the `OpponentDescriptor`
  (see `OpponentPool.sample()`), so the factory was never meant to handle that case.
- `OpponentPool(...)` now receives `self_checkpoint_manager=checkpoint_mgr` (reusing the
  same `SelfCheckpointManager` instance the trainer already checkpoints with — the
  class's own docstring specifies this exact pattern: *"so what this pool offers
  matches what actually got snapshotted"*) and `target_counts=config["opponent_pool"].get("full_run_targets")`.
  On the pilot config (no `full_run_targets` key), this falls back to `OpponentPool`'s
  even 3-way default harmlessly — the pilot still won't sample self-checkpoints during
  its warmup-only window, matching `phase4_pilot.yaml`'s documented, intentional scope.
  On the full-run/ablation configs, this now correctly enables the proportional
  self-checkpoint curriculum defined in `_shared_defaults.yaml` (`baseline1: 1667,
  baseline2: 1667, self_checkpoint: 1666` of 5000 episodes).

No changes were needed in `OpponentPool` or `SelfCheckpointManager` themselves — both
already fully supported this; the gap was only in how `run_phase4_pilot.py` wired them.

## 2. Known limitation to carry into the report — self-checkpoint match continuation

**What works after the fix:** during MCTS tree search, a sampled self-checkpoint
opponent behaves as its past self — `OpponentDescriptor.rollout_policy` is a
`MacroStrategyExecutorOpponent` proxy reconstructed from that checkpoint's compiled
prompt program, and `PyBulletSimulationBackend` runs it for all rollout simulations
regardless of opponent kind. This is the part that matters for SBSO's core mechanism
(learning to beat prior versions of itself via search).

**What does NOT yet work:** the *real, committed* match continuation — the physical
steps actually executed after a decision, via `match_trainer.py`'s
`env.opponent_policy = self.opponent_factory(opp_type)` — has no self-checkpoint case
in `_make_opponent_factory`. When `opp_type == "self_checkpoint"`, it silently falls
through to `make_rule_based_policy()`. So a self-checkpoint opponent plays the actual
match as a plain rule-based controller, not as a live re-enactment of its own past
strategy, even though MCTS *searched* against the correct proxy.

This is not a bug introduced by the fix above — it's a pre-existing, self-documented
gap. `src/sbso/opponent_pool.py`'s module docstring states it directly: *"a live-SLM
policy for the real, committed match continuation is a separate, not-yet-built hook."*

**Why this is a reasonable limitation to report rather than block on:**
- The rule-based fallback is a conservative substitute (never crashes, never gives an
  unfairly strong or degenerate opponent) rather than a silent correctness bug.
- The mismatch only affects the *execution* of a self-checkpoint-opponent episode, not
  the tree search that drives the strategy being learned — MCTS's evaluation of
  candidate branches remains checkpoint-accurate.
- Building the live-SLM continuation hook would require spinning up a second live
  inference path (a checkpoint-specific served model or adapter) mid-episode, which is
  a meaningfully larger engineering task than the fix above, not a quick follow-on.

**Suggested framing for the report's limitations section:** the self-checkpoint
opponent curriculum informs MCTS's search-time evaluation faithfully, but the
committed match continuation currently substitutes a rule-based policy in place of a
true self-play re-enactment; closing this gap is noted as future work rather than
addressed in the current SBSO implementation.

## ablation study 2000 episdes
# Reduced-Scope Ablation Study — Methodology Note

## Decision

The four Phase 5b ablation conditions (5b.1-5b.4: without SA, MCTS, DSPy, and
LLM-as-a-Judge respectively) are trained on **1,800 episodes each**, rather than the
full 5,000-episode schedule used for Benchmark 2. This does **not** affect Benchmark
2 itself, and does **not** affect RQ1/RQ2's headline claims — those are answered by
Phase 5a (Blocks B, C, D) against the full-scale Benchmark 2 checkpoint, which is
unchanged.

## Why this is defensible, not a shortcut

Per report.md 3.3.5.2, the ablation study's stated purpose is a per-component
**effect-size** signal (Cohen's d >= 0.3, "small-to-medium" threshold — deliberately
lower than the headline SBSO win-rate claim) answering *"whether one component
matters, not whether the whole system is competitive."* That is a directional,
comparative question against Benchmark 2, not an absolute-performance claim requiring
the same statistical power as the headline result. The existing episode counts in
`_shared_defaults.yaml` (5,000 total, K/W/delta recompilation triggers, MCTS
sim_budget) are themselves documented as placeholders pending pilot calibration, not
derived from a formal power analysis — so this reduction changes a placeholder
engineering choice, not an already-justified statistical design.

## What changed, concretely

Each ablation config now overrides two values from `_shared_defaults.yaml`, leaving
everything else (MCTS/DSPy/Judge hyperparameters, warmup schedule, checkpoint
interval) inherited unchanged — including the pilot-calibrated K/W/delta values once
those are frozen, since the ablations don't override that block:

| | Full run (Benchmark 2) | Ablations (5b.1-5b.4) |
|---|---|---|
| `episodes_total` | 5,000 | 1,800 |
| `opponent_pool.full_run_targets` | baseline1: 1667 / baseline2: 1667 / self_checkpoint: 1666 | baseline1: 600 / baseline2: 600 / self_checkpoint: 600 |
| `opponent_pool.warmup_episodes` | 500 (unchanged) | 500 (unchanged) |
| `self_checkpoint_interval_episodes` | 500 → ~10 checkpoints taken | 500 (unchanged) → ~3 checkpoints taken |

1,800 was chosen to stay comfortably above the fixed 500-episode warmup floor (during
which only baseline1/baseline2 opponents are eligible), leaving ~1,300 post-warmup
episodes for self-checkpoint opponents to actually accumulate a usable sample under
the proportional 600/600/600 split.

## Limitation to state explicitly in the report

The self-checkpoint curriculum each ablation trains against is thinner than
Benchmark 2's — roughly 3 self-checkpoint generations to play against, versus 10 on
the full run. If a specific ablation's degraded performance turns out to trace back to
having fewer, less-refined self-checkpoint opponents to train against (rather than the
ablated component itself), that's a confound this reduced scope introduces and should
be named as a limitation, not silently absorbed into the component's effect-size
result. Recommended framing for report.md's 3.6 Limitations section:

> "The four Phase 5b ablation conditions were trained on a reduced 1,800-episode
> schedule (versus Benchmark 2's 5,000) to manage compute cost, following the same
> warmup and opponent-pool proportions at smaller scale. This preserves the ablation
> study's directional effect-size comparison (Cohen's d >= 0.3 threshold) but yields a
> thinner self-checkpoint curriculum (~3 generations versus ~10) for each ablated
> variant, which is disclosed as a limitation on the precision of each component's
> isolated contribution."

## DSPy
k_rollout_batches: 25 — Trigger (a), the simple one: recompile every K episodes, on a fixed schedule, regardless of how training is going. At 25, that's a recompile attempt roughly every 25 episodes — 20 times across your 500-episode run.
rolling_window_w: 100 — The size of the sliding window used to compute a rolling average win rate (self._rolling_winrate(), the same function feeding your wandb rolling_winrate chart) — averaged over the last 100 episodes' win/loss outcomes.
reward_drop_threshold_delta: 0.05 — Trigger (b): compares the current rolling win rate against a recent baseline: if it's dropped by more than 0.05 (5 percentage points), that alone triggers an extra, off-schedule recompile attempt, independent of where you are in the K-episode cycle — a reactive "something's going wrong, try fixing the prompt now" mechanism.