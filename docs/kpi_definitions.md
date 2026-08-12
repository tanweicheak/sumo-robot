# KPI Definitions

Source of truth: report.md Sections 3.3.5.1.2, 3.3.5.3.2, 3.3.5.4, reconciled with the
implementation plan's clarification resolution log.

## Statistical test assignment (finalized)

| Metric | Primary hypothesis test | Absolute / effect-size reporting |
|---|---|---|
| Win rate (vs. Tier 1/2/3; B1 vs. B2) | Mann-Whitney U (across seed batches) | Percentage-point delta; Cohen's h available as secondary (not gating) |
| ELO rating | Mann-Whitney U (across independent tournament-run distributions) | within +/-100 ELO band |
| Decision latency | Bootstrap resampling | Cohen's d |
| Decision Stability Index (Phase 5c) | Mann-Whitney U | reported descriptively |
| Sensor Robustness Index (Phase 5c) | Bootstrap resampling | Cohen's d; reported descriptively, no fixed threshold |
| Schema validity rate under noise (Phase 5c) | Bootstrap resampling | Cohen's d |
| Latency compliance rate under noise (Phase 5c) | Bootstrap resampling | Cohen's d |

Win rate and ELO keep Mann-Whitney U as the primary test. Cohen's h (proportion effect
size) is implemented in `src/statistics/cohens_d.py` for secondary/robustness reporting
only; it does not gate any KPI.

## Phase 5a / 5b thresholds

| Metric | Threshold | Nature |
|---|---|---|
| Decision latency | <= 50ms | Pilot-calibrated design target (see note below) |
| Win rate vs. Tier 1 (rule-based) | >= 90% | Engineering sanity floor, diagnostic not evaluative |
| Win rate vs. Tier 2 (PPO) | >= 50% | Parity bar for "competitive" (RQ1) |
| Win rate vs. Tier 3 (Monolithic SLM) | >= 50% | Decomposition benefit (RQ2) |
| Win rate B1 vs. B2 | +10pp OR Cohen's d >= 0.5 | Headline SBSO improvement |
| ELO (various pairings) | within +/-100 points | Distribution-aware cross-check |
| Ablation effect size (per scenario, Phase 5b) | Cohen's d >= 0.3 | "small-to-medium" lower bar |

## Phase 5c thresholds

| Metric | Threshold | Nature |
|---|---|---|
| Schema validity rate under noise | >= 99% valid on first generation | Sanity bar (constrained decoding) |
| Latency compliance rate under noise | >= 95% of cycles within 50ms | Pilot-calibrated tolerance band |
| Sensor Robustness Index | none (descriptive) | Needs empirical noise scale first |
| Decision Stability Index | none (descriptive) | No natural baseline independent of pilot data |

## Note on the 50ms decision window

The 50ms figure is a **pilot-calibrated design target**, informed by but not equivalent
to the 42ms (sigma = 8ms) mean reaction time reported by Islam et al. (2026) for a
**microcontroller-based (STM32) hardware** sumo robot. That figure is an indicative
reaction-time scale for the domain, not a literature-proven requirement transferable to
an SLM reasoning pipeline. In the thesis, wherever 50ms is used as a KPI threshold,
state this epistemic status explicitly.

## Pilot-dependent values (not yet finalized)

These are config-driven and confirmed during pilot calibration, not hardcoded:
- Match count per pairing (default 500)
- DSPy recompilation cadence K, reward-drop threshold delta, rolling window W
- All Phase 5c thresholds pending characterization of real sensor-noise magnitude
- Empirical dataset size (target ceiling ~300 episodes / ~100 matches, TBC)
