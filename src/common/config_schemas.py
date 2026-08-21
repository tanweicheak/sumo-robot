"""
src.common.config_schemas

Phase: config-validation framework (Part 2 discussion)
Purpose: Per-phase Pydantic models validated against the resolved config dict at
    load time. Deliberately per-phase, not one global schema - each phase's config
    has a genuinely different shape, and one shared schema would either need
    everything Optional[Any] (defeating the point) or bloat into one file with
    fields that don't apply to most scripts.

    Two real bugs this session would have been caught here, at script-startup time
    (milliseconds), instead of silently propagating into a training run:
      - baseline2_ppo.yaml: shaping_push_weight: float = 0.05 (Python type-annotation
        syntax pasted into YAML, parses as the STRING "float = 0.05") - a type check
        on a float field catches this instantly.
      - _shared_defaults.yaml: dspy_recompilation.k_rollout_batches=5 vs. the
        scheduler's own documented intended default of 500 - a range/relationship
        check (see Phase4Config's cross-field validator) can flag large deviations.

    Deliberately does NOT change what load_config() returns (still a plain dict) -
    every existing `config["mcts"]["sim_budget"]`-style access across the codebase
    keeps working unchanged. validate_config() in config_loader.py just raises
    before returning if construction fails; nothing consumes the Pydantic model
    itself downstream. Adds zero new dependencies - pydantic is already required
    (see schemas.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class MCTSConfigSchema(BaseModel):
    sim_budget: int = Field(gt=0)
    horizon: int = Field(gt=0)
    judge_prune_threshold: float = Field(ge=0.0, le=1.0)


class DSPyRecompilationConfigSchema(BaseModel):
    k_rollout_batches: int = Field(gt=0)
    rolling_window_w: int = Field(gt=0)
    reward_drop_threshold_delta: float = Field(gt=0.0)


class Phase4Config(BaseModel):
    """Validated against config/phase4_*.yaml (pilot, full_sbso, ablation_no_*)."""

    model_config = {"extra": "allow"}   # config has many other keys (ablation, components,
                                         # checkpoint_output_dir, ...) not all worth a typed
                                         # field yet - this validates what's been added so
                                         # far without rejecting keys nobody's modeled.

    mcts: MCTSConfigSchema
    dspy_recompilation: DSPyRecompilationConfigSchema
    self_checkpoint_interval_episodes: int = Field(gt=0)
    episodes_total: int = Field(gt=0)

    @model_validator(mode="after")
    def _dspy_k_aligned_with_checkpoint_interval(self) -> "Phase4Config":
        # Soft check, not a hard reject - mirrors preflight_static_check.py's
        # dspy_k_vs_checkpoint_alignment WARN, encoded here as a hard validation
        # error instead, since this schema runs at every script's startup, not
        # just when preflight_static_check.py happens to be run separately.
        k = self.dspy_recompilation.k_rollout_batches
        interval = self.self_checkpoint_interval_episodes
        ratio = interval / k if k else float("inf")
        if ratio > 5:
            raise ValueError(
                f"dspy_recompilation.k_rollout_batches={k} but "
                f"self_checkpoint_interval_episodes={interval} (ratio {ratio:.0f}x). "
                "RecompilationScheduler's own docstring states the intended default is "
                "K=500, aligned with the self-checkpoint interval (ratio ~1x). If this "
                "is deliberate, raise the ratio threshold here rather than silently "
                "letting a >5x drift through."
            )
        return self


class PPOHyperparamsSchema(BaseModel):
    learning_rate: float = Field(gt=0.0, lt=1.0)
    n_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    n_epochs: int = Field(gt=0)
    gamma: float = Field(ge=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    clip_range: float = Field(gt=0.0)
    ent_coef: float = Field(ge=0.0)
    vf_coef: float = Field(gt=0.0)
    max_grad_norm: float = Field(gt=0.0)


class Phase1Config(BaseModel):
    """Validated against config/baseline2_ppo*.yaml."""

    model_config = {"extra": "allow"}

    ppo: PPOHyperparamsSchema
    train: dict


# Registry mapping build_run()'s `phase` string to the schema that applies, if any.
# A phase with no entry here simply isn't validated yet (phased rollout, not
# all-or-nothing) - config_loader.validate_config() treats a missing entry as a
# no-op, not an error.
PHASE_SCHEMAS: dict[str, type[BaseModel]] = {
    "phase4": Phase4Config,
    "phase4_pilot": Phase4Config,   # run_phase4_pilot.py - the real RunPod entrypoint this
                                     # schema exists to protect; was missing, silently no-op'd.
    "phase1": Phase1Config,
}