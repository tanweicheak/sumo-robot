"""
src.evaluation.match_runner

Phase: Phase 5a/5b evaluation (Phase 5c out of scope - see project decision to drop
    Hardware Emulation Layer / real-sensor-noise validation from this project's scope)
Purpose: The missing piece run_stress_test.py's own comment already named: turns a
    trained (or zero-shot) SLM checkpoint into an opponent_policy-shaped callable
    sumo_env.py can drive a match with. Does NOT introduce any new inference logic -
    every step below is a direct call into agents/schemas/clients that already exist
    and are already proven (perception_agent.py, opponent_analysis_agent.py,
    strategy_agent.py, tactical_execution_agent.py, actuator_bridge.py,
    sglang_server.py). This file is glue between two already-working things:
    graph_builder.py's real decision-cycle node functions and sumo_env.py's real
    OpponentPolicy contract - not a new implementation of either.

Benchmark 1 vs Benchmark 2 is NOT a code difference here - both go through the
identical SLMPolicyOpponent path. The only difference is which checkpoint the SGLang
agent server referenced by `sglang_agent_url` was launched with (base HF checkpoint
for Benchmark 1, SBSO-trained/LoRA-merged checkpoint for Benchmark 2) - this matches
Block D's own stated design (report.md 3.3.5.1: "both share identical multi-agent
topology and differ only in whether SBSO training was applied").

Usage:
    opponent = load_benchmark_opponent(
        benchmark="benchmark2", sglang_agent_url="http://localhost:30000",
    )
    env = PyBulletSumoEnv(..., opponent_policy=opponent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.agents.perception_agent import PerceptionAgent
from src.agents.opponent_analysis_agent import OpponentAnalysisAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.tactical_execution_agent import TacticalExecutionAgent
from src.agents.actuator_bridge import ActuatorBridge
from src.inference.sglang_server import SGLangSLMClient


@dataclass
class _MatchState:
    """Per-match state SLMPolicyOpponent must carry across calls within one episode -
    mirrors what graph_builder.py's real SumoRobotState tracks (lssd_history for OAA's
    multi-frame classification, prev_opponent_analysis for SA's pipelined-staleness
    read - see graph_builder.py's own docstring on why SA reads frame t-1, not t).
    Reset once per match via SLMPolicyOpponent.reset(), NOT on every call."""

    lssd_history: list[str] = field(default_factory=list)
    prev_opponent_analysis: object | None = None
    frame_index: int = 0


class SLMPolicyOpponent:
    """Wraps the real perception -> OAA -> SA -> TEA -> bridge decision cycle
    (graph_builder.py's actual node logic, called directly here rather than via a
    LangGraph StateGraph - this class IS the "dev/CI without a LangGraph install"
    path graph_builder.py's own docstring describes, applied to evaluation instead
    of live hardware) into the single-call shape sumo_env.py's OpponentPolicy
    already expects: (opp_obs: dict[str, np.ndarray]) -> (left_pwm, right_pwm).

    One instance = one opponent = one running SGLang server (agent checkpoint fixed
    at construction). Call .reset() between matches to clear lssd_history/
    prev_opponent_analysis - NOT safe to reuse across matches without resetting,
    since OAA's classification depends on accumulated history.
    """

    def __init__(
        self,
        perception_agent: PerceptionAgent,
        oaa: OpponentAnalysisAgent,
        sa: StrategyAgent,
        tea: TacticalExecutionAgent,
        bridge: ActuatorBridge,
    ) -> None:
        self.perception_agent = perception_agent
        self.oaa = oaa
        self.sa = sa
        self.tea = tea
        self.bridge = bridge
        self._state = _MatchState()

    def reset(self) -> None:
        """Call once per new match, before the first __call__. Clears accumulated
        lssd_history/prev_opponent_analysis - required because OAA's multi-frame
        classification and SA's pipelined-staleness read would otherwise leak state
        from a previous match into this one."""
        self._state = _MatchState()
        self.perception_agent.reset()

    def __call__(self, opp_obs: dict[str, np.ndarray]) -> tuple[float, float]:
        """opp_obs shape confirmed against sensors.py's real Sensors.read(): a dict
        with keys "tof", "ir", "encoder" (raw arrays, NOT pre-filtered - PerceptionAgent
        owns the Savitzky-Golay/IR-gradient/motor-deadband filtering internally, same
        as the live-hardware path)."""
        state = self._state

        # 1. Perception (deterministic, no SLM call) - same signature live inference
        #    already uses: perceive(raw_tof, raw_ir, raw_encoder).
        perception = self.perception_agent.perceive(
            opp_obs["tof"], opp_obs["ir"], opp_obs["encoder"],
        )
        state.lssd_history = state.lssd_history + [perception.lssd_text]

        # 2. OAA - classifies opponent behavior from accumulated LSSD history.
        opponent_analysis = self.oaa.analyze(state.lssd_history, state.frame_index)

        # 3. SA - pipelined staleness by design (graph_builder.py's own documented
        #    behavior, replicated here intentionally, not a bug): reads the analysis
        #    from the PREVIOUS frame, not the one just computed above. First call in
        #    a match has prev_opponent_analysis=None; StrategyAgent.decide must
        #    already tolerate that (same cold-start condition live inference hits
        #    on frame 0).
        macro = self.sa.decide(perception, state.prev_opponent_analysis)
        self._last_strategy = macro.strategy   # exposed for external logging -
                                                 # see scripts/run_phase5_eval.py
        state.prev_opponent_analysis = opponent_analysis

        # 4. TEA - macro strategy -> tactical command.
        command = self.tea.execute(perception, macro)

        # 5. Bridge - tactical command -> raw PWM, exactly what sumo_env.py's
        #    OpponentPolicy contract requires as this callable's return value.
        left_pwm, right_pwm = self.bridge.to_pwm(command)

        state.frame_index += 1
        return (left_pwm, right_pwm)


def load_benchmark_opponent(
    benchmark: Literal["benchmark1", "benchmark2"],
    sglang_agent_url: str,
    agent_model_path: str,
    *,
    perception_config_path: str | None = None,
    prompt_history_path: str | None = None,
) -> SLMPolicyOpponent:
    """The piece run_stress_test.py's own comment names as missing. Builds a real
    SLMPolicyOpponent the same way live inference builds its agent stack - one
    SGLangSLMClient shared across OAA/SA/TEA (matches how sglang_server.py's own
    docstring describes intended usage: point it at the HF directory used for
    LoRA fine-tuning, not the Phase 3 GGUF).

    `benchmark` controls one real thing, not just logging: whether StrategyAgent is
    constructed with a compiled DSPy prompt_program. Confirmed against
    strategy_agent.py's build_sa_prompt directly - `prompt_program=None` (the
    default) "reproduces the exact old behavior... so Benchmark 1 (zero-shot
    multi-agent) is unaffected" per that file's own comment. For benchmark2, the
    prompt_program MUST be supplied or StrategyAgent silently behaves exactly like
    Benchmark 1 (zero-shot) even though it's nominally being evaluated as the
    SBSO-trained condition - a real correctness bug, not a style choice, if left
    unset. Sourced from prompt_history.jsonl (the same file run_phase4_pilot.py's
    _on_episode_end already writes trainer.prompt_program to every episode) - reads
    the LAST entry, i.e. whatever prompt DSPy had compiled to by the end of training.

    Beyond that one distinction, benchmark1 vs benchmark2 is still purely which
    checkpoint `sglang_agent_url`'s server was launched against (see module
    docstring) - no other code branching.
    """
    client = SGLangSLMClient(server_url=sglang_agent_url, model_path=agent_model_path)

    prompt_program = None
    if benchmark == "benchmark2":
        if not prompt_history_path:
            raise ValueError(
                "load_benchmark_opponent(benchmark='benchmark2', ...) requires "
                "prompt_history_path - without it, StrategyAgent gets prompt_program="
                "None and silently evaluates identically to benchmark1 (zero-shot). "
                "Point this at the trained variant's prompt_history.jsonl, e.g. "
                "checkpoints/benchmark2_full_sbso/prompt_history.jsonl."
            )
        prompt_program = _load_final_prompt_program(prompt_history_path)

    perception_agent = PerceptionAgent(config_path=perception_config_path)
    oaa = OpponentAnalysisAgent(client=client)
    sa = StrategyAgent(client=client, prompt_program=prompt_program)
    tea = TacticalExecutionAgent(client=client)
    bridge = ActuatorBridge()

    return SLMPolicyOpponent(
        perception_agent=perception_agent, oaa=oaa, sa=sa, tea=tea, bridge=bridge,
    )


def _load_final_prompt_program(prompt_history_path: str) -> str:
    """CORRECTED: prompt_history.jsonl's real schema (confirmed against actual
    written data) is {episode, trigger_reason, prompt_version, rolling_winrate,
    accepted, validation} - it never contains a prompt_program field at all, despite
    this function's original docstring claiming otherwise. The real compiled prompt
    text is written to progress.json instead (format: {"episode": N,
    "prompt_program": "SA_PROMPT_VN\n...", ...}), but only periodically - at
    self_checkpoint_interval_episodes boundaries and episode 0 - NOT every episode.
    This function now reads progress.json and returns its prompt_program value,
    which is the LAST periodic snapshot taken, not necessarily the exact final
    episode's prompt if training continued past the last snapshot before ending.
    Despite the parameter name (kept for call-site compatibility), pass the
    checkpoint's progress.json path here, not prompt_history.jsonl's path."""
    import json
    from pathlib import Path

    path = Path(prompt_history_path)
    if not path.exists():
        raise FileNotFoundError(f"progress.json not found at {path}")

    with path.open() as f:
        data = json.load(f)
    program = data.get("prompt_program")
    if not program:
        raise ValueError(
            f"{path} has no prompt_program value - training may not have reached "
            "its first checkpoint interval, or this is the wrong file."
        )
    return program


def _load_final_prompt_program_OLD_UNUSED(prompt_history_path: str) -> str:
    """Superseded - see _load_final_prompt_program above. Kept only so the
    remainder of this file's original loop-based parsing code doesn't need
    re-indenting; never called."""
    import json
    from pathlib import Path

    path = Path(prompt_history_path)
    if not path.exists():
        raise FileNotFoundError(f"prompt_history.jsonl not found at {path}")

    last_program: str | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "prompt_program" in entry and entry["prompt_program"] is not None:
                last_program = entry["prompt_program"]

    if last_program is None:
        raise ValueError(
            f"{path} has no non-null prompt_program entries - training may not have "
            "run long enough for a DSPy recompile to fire, or this is the wrong file."
        )
    return last_program