"""
src.sbso.dspy_compiler

Phase: Phase 4
Purpose: DSPy prompt-program recompilation (report Section 3.3.4). Optimizes the SA
    prompt using MCTS-discovered high-reward (state -> strategy) pairs as the training
    set, with MCTS value as the metric (Branch 8/Q13). MockDSPyCompiler validates the
    loop; the real DSPy teleprompter lands in Stage 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DSPyCompiler(ABC):
    @abstractmethod
    def compile(self, training_pairs: list, current_prompt: str) -> str:
        ...


class MockDSPyCompiler(DSPyCompiler):
    def __init__(self) -> None:
        self.compile_count = 0
        self.last_dominant_strategy = None   # mock has no structured demos to compute this from

    def compile(self, training_pairs: list, current_prompt: str) -> str:
        self.compile_count += 1
        # Real DSPy selects/refines few-shot examples from training_pairs; the mock just
        # marks that a recompilation happened so the loop's cadence is observable.
        return f"{current_prompt}#recompiled_v{self.compile_count}"

"""
(append to src/sbso/dspy_compiler.py)

RealDSPyCompiler - wraps the project's llama.cpp backend as a dspy.LM so DSPy optimizes
few-shot examples for the SAME model TEA/SA run at inference (not an OpenAI-style API).
Metric = exact-match against MCTS's chosen strategy (Branch 8/Q13: MCTS value drove which
pairs are "good"; DSPy's job is teaching the SLM to reproduce that choice from state alone).
"""


class RealDSPyCompiler(DSPyCompiler):
    def __init__(
        self,
        llama_cpp_api_base: str | None = None,
        sglang_api_base: str | None = None,
        llama_model_path: str | None = None,   # metadata/logging only - see _ensure_lm
        n_ctx: int = 2048,
        max_bootstrapped_demos: int = 4,
    ) -> None:
        if not llama_cpp_api_base and not sglang_api_base:
            raise ValueError("RealDSPyCompiler needs sglang_api_base or llama_cpp_api_base")
        self.llama_cpp_api_base = llama_cpp_api_base
        self.sglang_api_base = sglang_api_base
        self.llama_model_path = llama_model_path
        self.n_ctx = n_ctx
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.compile_count = 0
        self._lm = None
        self.last_dominant_strategy = None   # set by _extract_prompt_program each compile()

    def _ensure_lm(self):
        if self._lm is None:
            import dspy

            if self.sglang_api_base:
                # SGLang's OpenAI-compatible endpoint; api_key is unchecked by SGLang
                # but litellm/dspy require a non-empty string.
                self._lm = dspy.LM(
                    model="openai/sglang-agent", api_base=self.sglang_api_base, api_key="EMPTY",
                )
            else:
                # D5b fix: "llama_cpp/" was never a real LiteLLM provider prefix - LiteLLM
                # (which dspy.LM routes through) only talks to models over HTTP, via
                # network-based providers (OpenAI-compatible endpoints, Ollama, etc). It
                # cannot load llama-cpp-python in-process, so the old code would have
                # failed at runtime with a LiteLLM "provider not found" error the first
                # time compile() actually ran. Fix: talk to llama.cpp's own
                # OpenAI-compatible HTTP server instead - same pattern as the sglang
                # branch above. Start one with either:
                #   python -m llama_cpp.server --model <path/to.gguf> --port 8080
                # or llama.cpp's own `llama-server -m <path/to.gguf> --port 8080`
                # then pass llama_cpp_api_base="http://127.0.0.1:8080/v1".
                self._lm = dspy.LM(
                    model="openai/local-llama", api_base=self.llama_cpp_api_base, api_key="EMPTY",
                )
            dspy.configure(lm=self._lm)
        return self._lm

    def _select_examples(self, training_pairs: list, max_matches: int = 10, max_total: int = 50) -> list:
        """Match-aware selection (fix for the DSPy few-shot recency-window issue): pick
        pairs from the most recent `max_matches` distinct episodes, capped at
        `max_total` overall, spread roughly evenly across those matches - NOT a flat
        training_pairs[-max_total:] tail. Now that one episode = one full match (many
        decisions), a flat tail slice could come entirely from the single most recent
        match/opponent, losing the cross-opponent diversity DSPy's few-shot examples
        need. Expects training_pairs entries as (episode, state, strategy) triples;
        falls back to treating entries as (state, strategy) pairs from a single
        pseudo-episode if untagged, for compatibility with Stage 1/2 callers that don't
        tag by episode."""
        if not training_pairs:
            return []

        by_episode: dict = {}
        order: list = []
        for entry in training_pairs:
            if len(entry) == 3:
                ep, state, strategy = entry
            else:
                ep, (state, strategy) = 0, entry   # untagged (Stage 1/2) - single bucket
            if ep not in by_episode:
                by_episode[ep] = []
                order.append(ep)
            by_episode[ep].append((state, strategy))

        recent_episodes = order[-max_matches:]
        per_match_cap = max(1, max_total // max(1, len(recent_episodes)))

        selected = []
        for ep in recent_episodes:
            selected.extend(by_episode[ep][-per_match_cap:])
        return selected[-max_total:]

    def compile(self, training_pairs: list, current_prompt: str) -> str:
        import dspy

        self._ensure_lm()
        self.compile_count += 1

        class SAStrategySignature(dspy.Signature):
            """Select the macro strategy for a sumo robot given its LSSD state."""
            state_desc: str = dspy.InputField(desc="LSSD semantic state text")
            strategy: str = dspy.OutputField(desc="one of: charge, flank, retreat, hold, evade_edge")

        examples = []
        for state, strategy in self._select_examples(training_pairs):
            state_desc = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
            strat_value = strategy.value if hasattr(strategy, "value") else str(strategy)
            examples.append(dspy.Example(state_desc=state_desc, strategy=strat_value).with_inputs("state_desc"))

        if not examples:
            return current_prompt

        predictor = dspy.Predict(SAStrategySignature)

        def _metric(example, prediction, trace=None):
            return float(example.strategy == prediction.strategy)

        # DIAGNOSTIC: num_threads=1 added to isolate whether the "input_ids should be a
        # list of lists for batch processing" error from SGLang's OpenAI-compatible
        # endpoint is triggered by BootstrapFewShot's default concurrent evaluation
        # (never previously exercised - all other calls in this pipeline go through
        # SGLang's native /generate endpoint via a proven-working ThreadPoolExecutor,
        # not this OpenAI-compatible /v1/chat/completions path). If this resolves the
        # error, the root cause is concurrency-specific on SGLang's OpenAI-compat layer
        # for this sglang==0.4.6.post5 pin - re-evaluate before removing this, don't
        # just delete it once it starts working.
        teleprompter = dspy.BootstrapFewShot(
            metric=_metric, max_bootstrapped_demos=self.max_bootstrapped_demos, num_threads=1,
        )
        compiled = teleprompter.compile(predictor, trainset=examples)
        return self._extract_prompt_program(compiled, current_prompt)

    def _extract_prompt_program(self, compiled_module, fallback_prompt: str) -> str:
        try:
            demos = compiled_module.demos
            lines = [f"{d.state_desc} -> {d.strategy}" for d in demos]
            # Item 5 fix: compute the dominant strategy HERE, at the source, from
            # structured demo objects - not via fragile substring-counting on the
            # rendered text later (opponent_pool.py's old _strategy_from_prompt_program).
            self.last_dominant_strategy = self._dominant_strategy(demos)
            return f"SA_PROMPT_V{self.compile_count}\n" + "\n".join(lines)
        except Exception:
            return fallback_prompt

    @staticmethod
    def _dominant_strategy(demos):
        from collections import Counter
        if not demos:
            return None
        counts = Counter(d.strategy for d in demos)
        return counts.most_common(1)[0][0]   # plain string, e.g. "charge" - matches MacroStrategy.value


def validate_prompt_candidate(
    candidate_prompt: str,
    incumbent_prompt: str,
    training_pairs: list,
    slm_client,
    judge,
    n_samples: int = 30,
    accept_margin: float = 0.0,
    seed: int = 0,
) -> dict:
    """State-replay validation for a newly-compiled DSPy prompt, before committing it
    live. Deliberately does NOT play new match episodes (that would inflate episode
    count past the planned total) - instead re-runs SA inference on a small sample of
    ALREADY-COLLECTED states from training_pairs (no new physics), under both the
    incumbent and candidate prompt, and compares mean Judge score. Only the two
    prompts and a handful of extra inference calls are new; no new PyBullet episodes
    are involved anywhere in this function.

    Returns {"accept": bool, "incumbent_mean_score": float, "candidate_mean_score": float,
    "n_samples": int} - caller (match_trainer.py's recompile block) decides whether to
    replace self.prompt_program based on "accept".
    """
    import random

    from src.agents.schemas import DirectionLabel, DistanceLabel, EdgeLabel, MomentumLabel, PerceptionState
    from src.agents.strategy_agent import StrategyAgent

    rng = random.Random(seed)
    sample = rng.sample(training_pairs, min(n_samples, len(training_pairs)))

    incumbent_agent = StrategyAgent(client=slm_client, prompt_program=incumbent_prompt)
    candidate_agent = StrategyAgent(client=slm_client, prompt_program=candidate_prompt)

    incumbent_scores, candidate_scores = [], []
    for _episode, state, _strategy in sample:
        lssd_text = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
        # Only lssd_text is actually read by build_sa_prompt() - the other
        # PerceptionState fields are structurally required but not consulted for
        # this prompt, so neutral placeholders here don't affect the comparison.
        perception = PerceptionState(
            lssd_text=lssd_text, opp_distance=DistanceLabel.MID, opp_direction=DirectionLabel.CENTER,
            edge=EdgeLabel.SAFE, momentum=MomentumLabel.STABLE, opp_distance_m=1.0,
        )
        incumbent_decision = incumbent_agent.decide(perception, prev_oaa=None)
        candidate_decision = candidate_agent.decide(perception, prev_oaa=None)
        incumbent_scores.append(judge.score_branch(lssd_text, incumbent_decision.strategy))
        candidate_scores.append(judge.score_branch(lssd_text, candidate_decision.strategy))

    incumbent_mean = sum(incumbent_scores) / len(incumbent_scores) if incumbent_scores else 0.0
    candidate_mean = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0

    return {
        "accept": candidate_mean >= incumbent_mean - accept_margin,
        "incumbent_mean_score": round(incumbent_mean, 4),
        "candidate_mean_score": round(candidate_mean, 4),
        "n_samples": len(sample),
    }