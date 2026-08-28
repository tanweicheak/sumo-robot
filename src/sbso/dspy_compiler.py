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


def _make_sglang_native_lm_class():
    """Factory that defines SGLangNativeLM as a real dspy.BaseLM subclass, only at
    call time - keeps `import dspy` lazy (this file's existing convention: dspy/
    litellm are slow to import, and MockDSPyCompiler-only callers shouldn't pay
    that cost - see class docstring below for why this adapter exists at all)."""
    import dspy

    class SGLangNativeLM(dspy.BaseLM):
        """dspy.BaseLM subclass that routes through SGLang's native /generate
        endpoint instead of its OpenAI-compatible /v1/chat/completions endpoint.

        Why this exists: this SGLang server (sglang==0.4.6.post5) returns
        'input_ids should be a list of lists for batch processing' on EVERY
        /v1/chat/completions request, unconditionally - confirmed via a bare
        openai-client repro with no DSPy/litellm involved (4/4 test shapes
        failed identically), and confirmed NOT a chat-template issue
        (Phi-4-mini's own real template fails with the identical error; a
        wrong generic 'chatml' template merely avoids the crash while
        producing incorrect output - see session notes). Matches the same
        error string as sgl-project/sglang issue #25593 (different
        model/endpoint/version) - unresolved upstream as of this writing, no
        available workaround via launch flags. The native /generate endpoint
        is a separate code path in SGLang that has been reliable all session
        (episode collection, MCTS, the constrained-decoding preflight check) -
        this adapter reuses it instead of waiting on an upstream fix.

        forward_contract="legacy" (DSPy's default): forward(prompt=None,
        messages=None, **kwargs) must satisfy BaseLM._process_completion's and
        _process_lm_response's actual attribute-access contract - confirmed
        against the real installed dspy==3.3.1 source (not assumed from
        possibly-mismatched docs): response.choices (each choice a dict with
        a "text" key is explicitly supported via _process_completion's own
        c["text"] fallback branch), response.model, and
        getattr(response, "usage"/"cache_hit"/"_hidden_params", ...). A plain
        dict fails this (dict.choices != dict["choices"]) - returns a
        types.SimpleNamespace instead.

        Single-turn text completion only (confirmed sufficient for this
        project's SAStrategySignature/_metric use in RealDSPyCompiler.compile)
        - does NOT implement true multi-turn chat history threading. If
        `messages` contains more than one message, they're concatenated into
        a single prompt string with role labels, not sent as a structured
        conversation - fine for this project's usage, would need extending
        for real multi-turn.
        """

        def __init__(
            self,
            server_url: str,
            model_path: str,
            model: str = "sglang-native",
            temperature: float = 0.0,
            max_tokens: int = 150,
            timeout_s: float = 30.0,
            **kwargs,
        ) -> None:
            super().__init__(
                model=model, model_type="chat", temperature=temperature,
                max_tokens=max_tokens, cache=False, **kwargs,
            )
            self.server_url = server_url.rstrip("/")
            self._model_path = model_path
            self._tokenizer = None   # lazy-loaded in _chat_format, not per-call
            self.timeout_s = timeout_s
            self.call_count = 0

        def _chat_format(self, prompt: str) -> str:
            """Same fix as src/inference/sglang_server.py's SGLangSLMClient - wraps
            prompt in the model's real chat template instead of sending raw
            concatenated text. Applied on top of _flatten_prompt's existing
            multi-message concatenation, not replacing it - that logic still handles
            turning a `messages` list into one string; this wraps the RESULT in a
            real chat turn before it goes to SGLang's native /generate endpoint."""
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )

        def _flatten_prompt(self, prompt: str | None, messages: list[dict] | None) -> str:
            if prompt is not None:
                return prompt
            if messages:
                return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
            return ""

        def _json_constraint_regex(self, messages: list[dict] | None) -> str | None:
            """Detects DSPy's JSONAdapter-style prompt (the "Outputs will be a JSON
            object" marker, confirmed present in the real system message via debug
            logging) and returns a regex anchoring the whole response to valid JSON
            with `strategy` constrained to MacroStrategy's real enum values - reusing
            enum_regex_pattern's existing longest-first alternation logic rather than
            duplicating it (sglang_server.py already relies on that ordering to avoid
            a short value like "charge" short-matching inside "charge_forward"-style
            longer values).

            Why this exists: unconstrained JSON prompting was NOT reliable for this
            model - real pilot logs showed 'No other text or explanations would be
            accepted. `strategy`: "hold"', '{camelCase}', and (once max_tokens was
            raised) truncated free-association like '"tik-tak-tik-tak-tik-t"' cut off
            at the token limit. xgrammar (confirmed via /get_server_info as this
            server's grammar_backend) enforces this at decode time - not a prompted
            request the model can ignore, the same hard guarantee already proven
            reliable for the live-agent/judge regex constraints all session.

            Returns None for non-JSON prompts (e.g. ChatAdapter's first-attempt
            `[[ ## field ## ]]` format, which succeeded unconstrained in earlier
            testing) - applying a JSON-shaped regex there would break it, not help.
            Currently hardcoded to the `strategy` field/MacroStrategy - this project's
            only real DSPy use case (SAStrategySignature) - would need generalizing if
            RealDSPyCompiler is ever pointed at a signature with a different output
            field or enum.
            """
            if not messages:
                return None
            combined = " ".join(m.get("content", "") for m in messages)
            if "Outputs will be a JSON object" not in combined:
                return None

            from src.inference.grammar import enum_regex_pattern
            from src.agents.schemas import MacroStrategy

            values = [s.value for s in MacroStrategy]
            enum_alt = enum_regex_pattern(values)
            # Whitespace-tolerant JSON object regex: {"strategy": "<one of the enum
            # values>"} allowing SGLang/xgrammar's own whitespace handling around the
            # colon and braces, matching the shape JSONAdapter's prompt asks for.
            return r'\{\s*"strategy"\s*:\s*"' + enum_alt + r'"\s*\}'

        def forward(self, prompt: str | None = None, messages: list[dict] | None = None, **kwargs):
            import requests
            from types import SimpleNamespace

            text_prompt = self._chat_format(self._flatten_prompt(prompt, messages))
            sampling_params = {
                "temperature": kwargs.get("temperature", self.kwargs.get("temperature", 0.0)) or 0.0,
                # 150, not 8 (sglang_server.py's live-agent default): confirmed via
                # debug logging that unconstrained DSPy calls need real room to work
                # through JSONAdapter/ChatAdapter's formatting instructions before
                # producing an actual answer - a small model with no few-shot demos
                # yet was observed echoing/paraphrasing the instructions themselves
                # ("Do not include any text outside of the...") and getting cut off
                # mid-sentence at 8 tokens, never reaching the answer. The live-agent
                # path (sglang_server.py) can stay at 8 because it constrains output
                # via a regex to a single enum word - this path has no such
                # constraint and needs the extra budget instead.
                "max_new_tokens": kwargs.get("max_tokens", self.kwargs.get("max_tokens", 150)),
            }
            json_regex = self._json_constraint_regex(messages)
            if json_regex is not None:
                sampling_params["regex"] = json_regex

            payload = {"text": text_prompt, "sampling_params": sampling_params}
            self.call_count += 1
            resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            completion_text = (data.get("text") or "").strip()
            meta = data.get("meta_info", {}) or {}

            # BaseLM._process_lm_response/_process_completion (confirmed against the
            # actual installed dspy==3.3.1 source, not assumed) does ATTRIBUTE access -
            # response.choices, response.model, getattr(response, "usage"/"cache_hit"/
            # "_hidden_params") - a plain dict fails these (dict.choices is not
            # dict["choices"]). SimpleNamespace for the top-level response satisfies
            # every confirmed access. Each choice item stays a plain dict on purpose:
            # _process_completion's own code has an explicit c["text"] fallback branch
            # for exactly this case (no .message attribute -> dict key instead), so a
            # nested message/role object isn't needed here.
            return SimpleNamespace(
                choices=[{"text": completion_text}],
                model=self.model,
                usage={
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "total_tokens": meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0),
                },
                cache_hit=False,
                _hidden_params={},
            )

    return SGLangNativeLM


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
                # Was: dspy.LM(model="openai/sglang-agent", api_base=self.sglang_api_base,
                # api_key="EMPTY") - routed through SGLang's OpenAI-compatible endpoint.
                # Confirmed broken: /v1/chat/completions returns "input_ids should be a
                # list of lists for batch processing" on EVERY request, unconditionally
                # (bare openai-client repro, no DSPy/litellm involved, 4/4 shapes failed
                # identically) - see SGLangNativeLM's docstring for the full investigation.
                # Fix: use the native /generate endpoint instead, via a proper
                # dspy.BaseLM subclass. sglang_api_base was built as f"{agent_server_url}/v1"
                # for the old OpenAI-compat path - strip that suffix, SGLangNativeLM talks
                # to {server_url}/generate directly.
                native_base = self.sglang_api_base
                if native_base.endswith("/v1"):
                    native_base = native_base[: -len("/v1")]
                SGLangNativeLM = _make_sglang_native_lm_class()
                # llama_model_path was previously "metadata/logging only" per this
                # constructor's own comment - now genuinely required, for the chat
                # template fix (SGLangNativeLM needs the real HF directory to load
                # a tokenizer). If this is None, the chat-template fix silently
                # cannot apply here - raise clearly instead of a confusing failure
                # three calls deep inside _chat_format.
                if not self.llama_model_path:
                    raise ValueError(
                        "RealDSPyCompiler(sglang_api_base=..., llama_model_path=...) "
                        "now requires llama_model_path - it's used to load the real "
                        "tokenizer for chat-template formatting, not just metadata. "
                        "Pass the agent's HF model directory (same one used to launch "
                        "its SGLang server)."
                    )
                self._lm = SGLangNativeLM(
                    server_url=native_base, model="sglang-agent", model_path=self.llama_model_path,
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
                # (llama.cpp's own OpenAI-compat server is a different codebase than
                # SGLang's - no evidence it shares this bug, left as dspy.LM/litellm here.)
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

    def compile(self, training_pairs: list, current_prompt: str, episode_outcomes: dict | None = None) -> str:
        import dspy

        self._ensure_lm()
        self.compile_count += 1

        # Outcome-aware filter: only use examples from WON episodes, not every
        # episode regardless of outcome. Real, evidenced fix for the strategic
        # lock-in finding - DSPy was previously trained to imitate whatever MCTS
        # chose regardless of win/loss/draw. Backward compatible: falls back to the
        # full, unfiltered training_pairs if episode_outcomes is empty (e.g. no
        # wins yet early in training) or not passed at all.
        if episode_outcomes:
            won_episodes = {ep for ep, val in episode_outcomes.items() if val == 1.0}
            filtered = [
                entry for entry in training_pairs
                if len(entry) == 3 and entry[0] in won_episodes
            ]
            if filtered:
                training_pairs = filtered
            # else: no won episodes yet - proceed with the full, unfiltered set

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

        teleprompter = dspy.BootstrapFewShot(metric=_metric, max_bootstrapped_demos=self.max_bootstrapped_demos)
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
            lssd_text=lssd_text, opp_distance=DistanceLabel.MID, opp_direction=DirectionLabel.FRONT_CENTER,
            edge=EdgeLabel.SAFE, momentum=MomentumLabel.STILL, opp_distance_m=1.0,
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