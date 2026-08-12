"""
src.sbso.judge

Phase: Phase 4
Purpose: LLM-as-a-Judge (report Section 3.3.4.1). Scores tactical positions/branches to
    (1) prune low-quality MCTS branches before simulation and (2) estimate leaf values in
    truncated rollouts. MockJudge validates the loop locally; the real Llama-3.1-8B judge
    (Phase 4 cloud) implements the same interface. The Judge runs ONLY during offline
    training - never quantized, exported, or deployed.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class Judge(ABC):
    @abstractmethod
    def score_position(self, state_desc: str) -> float:
        """Estimate how winning a position is, in [0, 1]. Used as the leaf value in
        truncated MCTS rollouts."""

    @abstractmethod
    def score_branch(self, state_desc: str, strategy) -> float:
        """Score the tactical coherence of taking `strategy` from this state, in [0, 1].
        Used to prune low-quality branches before spending simulation budget."""


class MockJudge(Judge):
    """Deterministic-by-seed stand-in. Returns plausible scores so the SBSO loop runs
    locally in milliseconds without a real model."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.call_count = 0

    def score_position(self, state_desc: str) -> float:
        self.call_count += 1
        return self._rng.uniform(0.0, 1.0)

    def score_branch(self, state_desc: str, strategy) -> float:
        self.call_count += 1
        return self._rng.uniform(0.3, 1.0)

"""
(append to src/sbso/judge.py)

LlamaCppJudge - real Llama-3.1-8B-Instruct judge via llama.cpp. Discretizes scoring to
a single grammar-constrained digit (1-5), mapped to [0,1] - same speed rationale as
TEA's single-token GBNF output (Phase 3): one constrained token is far cheaper than
free-form reasoning text, and grammar-guarantees a parseable score every time.
"""

from pathlib import Path


class LlamaCppJudge(Judge):
    _SCALE = ["1", "2", "3", "4", "5"]

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 1024,
        n_gpu_layers: int = -1,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose
        self._llm = None
        self._grammar = None
        self.call_count = 0

    def _ensure_loaded(self):
        if self._llm is None:
            from llama_cpp import Llama, LlamaGrammar

            self._llm = Llama(
                model_path=self.model_path, n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers, seed=self.seed, verbose=self.verbose,
            )
            self._grammar = LlamaGrammar.from_string('root ::= "1" | "2" | "3" | "4" | "5"\n')
        return self._llm

    def _score_prompt(self, prompt: str) -> float:
        self.call_count += 1
        llm = self._ensure_loaded()
        out = llm.create_completion(prompt=prompt, grammar=self._grammar, max_tokens=2, temperature=0.0)
        text = out["choices"][0]["text"].strip()
        try:
            digit = int(text)
        except ValueError:
            digit = 3   # neutral fallback; grammar should prevent this
        digit = max(1, min(5, digit))
        return (digit - 1) / 4.0   # map 1..5 -> 0.0..1.0

    def score_position(self, state_desc: str) -> float:
        prompt = (
            "You are judging a sumo robot match position. Rate how winning this "
            "position looks for the acting robot, 1 (losing badly) to 5 (dominant).\n"
            f"Position: {state_desc}\nRating (1-5):"
        )
        return self._score_prompt(prompt)

    def score_branch(self, state_desc: str, strategy) -> float:
        strat_name = strategy.value if hasattr(strategy, "value") else str(strategy)
        prompt = (
            "You are judging tactical coherence of a proposed sumo strategy. Rate how "
            "coherent this choice is given the state, 1 (nonsensical) to 5 (clearly correct).\n"
            f"State: {state_desc}\nProposed strategy: {strat_name}\nRating (1-5):"
        )
        return self._score_prompt(prompt)

"""
(append to src/sbso/judge.py)

SGLangJudge - real Judge served via SGLang. Same digit-discretized scoring as
LlamaCppJudge, plus score_branches_batch for concurrent multi-branch pruning (MCTS
scores up to 5 branches per node - firing them concurrently is where SGLang's batching
actually pays off versus one-at-a-time llama.cpp calls).
"""

from concurrent.futures import ThreadPoolExecutor


class SGLangJudge(Judge):
    def __init__(self, server_url: str, temperature: float = 0.0, timeout_s: float = 30.0,
                 max_concurrency: int = 16) -> None:
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self.call_count = 0

    def _score_prompt(self, prompt: str) -> float:
        import requests

        self.call_count += 1
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": self.temperature, "max_new_tokens": 2,
                                "regex": "(1|2|3|4|5)"},
        }
        resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        text = (resp.json().get("text") or "").strip()
        try:
            digit = int(text)
        except ValueError:
            digit = 3
        return (max(1, min(5, digit)) - 1) / 4.0

    def score_position(self, state_desc: str) -> float:
        prompt = (
            "You are judging a sumo robot match position. Rate how winning this "
            "position looks for the acting robot, 1 (losing badly) to 5 (dominant).\n"
            f"Position: {state_desc}\nRating (1-5):"
        )
        return self._score_prompt(prompt)

    def score_branch(self, state_desc: str, strategy) -> float:
        strat_name = strategy.value if hasattr(strategy, "value") else str(strategy)
        prompt = (
            "You are judging tactical coherence of a proposed sumo strategy. Rate how "
            "coherent this choice is given the state, 1 (nonsensical) to 5 (clearly correct).\n"
            f"State: {state_desc}\nProposed strategy: {strat_name}\nRating (1-5):"
        )
        return self._score_prompt(prompt)

    def score_branches_batch(self, state_desc: str, strategies: list) -> list[float]:
        """Score all candidate branches at an MCTS node CONCURRENTLY. This is the
        pre-rollout pruning step (5 branches per node) - the single highest-value place
        to use SGLang's batching, since it happens at every node expansion."""
        results: list[float] = [0.0] * len(strategies)

        def _one(i: int, strategy) -> None:
            results[i] = self.score_branch(state_desc, strategy)

        workers = min(self.max_concurrency, max(1, len(strategies)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_one, i, s) for i, s in enumerate(strategies)]
            for f in futures:
                f.result()
        return results