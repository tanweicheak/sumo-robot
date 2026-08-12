"""
src.sbso.mcts

Phase: Phase 4
Purpose: Monte Carlo Tree Search over MacroStrategy sequences (report Section 3.3.4,
    "Explore Macrointent"). Node = sim state (LSSD + opponent classification embedded);
    branch = one of SA's 5 macro-strategies. The four steps: Selection (UCT descent),
    Expansion (add an untried strategy child), Simulation (backend rollout -> Judge value),
    Backpropagation (update ancestors). Returns the best strategy for the root state plus
    the (state -> best_strategy) training pair that DSPy/LoRA later distill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class MCTSNode:
    __slots__ = ("state", "parent", "strategy_from_parent", "children", "untried",
                 "visits", "value_sum")

    def __init__(self, state, parent=None, strategy_from_parent=None, untried=None):
        self.state = state
        self.parent = parent
        self.strategy_from_parent = strategy_from_parent
        self.children: dict = {}
        self.untried: list = list(untried) if untried else []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def best_child(self, c_uct: float) -> "MCTSNode":
        best, best_score = None, -1e18
        for child in self.children.values():
            if child.visits == 0:
                score = 1e17   # force-visit unvisited children first
            else:
                exploit = child.value_sum / child.visits
                explore = c_uct * math.sqrt(math.log(self.visits + 1) / child.visits)
                score = exploit + explore
            if score > best_score:
                best, best_score = child, score
        return best


@dataclass
class MCTSResult:
    best_strategy: Any
    root_stats: dict
    training_pair: tuple


class MCTS:
    def __init__(self, backend, strategies, c_uct: float = 1.41, sim_budget: int = 30,
                 horizon: int = 4, judge_prune_threshold: float = 0.0) -> None:
        self.backend = backend
        self.strategies = list(strategies)
        self.c_uct = c_uct
        self.sim_budget = sim_budget
        self.horizon = horizon
        self.judge_prune_threshold = judge_prune_threshold

    def search(self, root_state) -> MCTSResult:
        root = MCTSNode(root_state, untried=self._legal(root_state))
        for _ in range(self.sim_budget):
            node = self._select(root)
            node = self._expand(node)
            value = self.backend.rollout(node.state, self.horizon)
            self._backprop(node, value)

        best = self._best_by_mean(root)
        stats = {
            (s.value if hasattr(s, "value") else str(s)):
            (round(root.children[s].mean_value, 4) if s in root.children else None)
            for s in self.strategies
        }

        # D3 fix: the tree (root + every expanded child) is discarded once we return -
        # free any backend-owned per-search resources it created (PyBulletSimulationBackend
        # saveState ids; a no-op for MockSimulationBackend, which has no such method).
        # The caller's root_state id is kept: it still needs it to restore the live env.
        release = getattr(self.backend, "release_search_states", None)
        if release is not None:
            keep_id = getattr(root_state, "pybullet_state_id", None)
            release(keep={keep_id} if keep_id is not None else None)
            
        return MCTSResult(best_strategy=best, root_stats=stats, training_pair=(root_state, best))

    def _legal(self, state) -> list:
        """Judge pre-pruning: drop branches the Judge scores below threshold before they
        consume simulation budget. Uses batched concurrent scoring when the backend's
        judge supports it (SGLangJudge), falling back to sequential otherwise."""
        legal = list(self.strategies)
        if self.judge_prune_threshold <= 0.0:
            return legal

        judge = getattr(self.backend, "judge", None)
        if judge is not None and hasattr(judge, "score_branches_batch"):
            scores = judge.score_branches_batch(
                state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", ""),
                legal,
            )
            kept = [s for s, sc in zip(legal, scores) if sc >= self.judge_prune_threshold]
        else:
            kept = [s for s in legal
                    if self.backend.judge_branch(state, s) >= self.judge_prune_threshold]
        return kept if kept else list(self.strategies)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.c_uct)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried:
            return node
        strat = node.untried.pop()
        next_state = self.backend.step(node.state, strat)
        child = MCTSNode(next_state, parent=node, strategy_from_parent=strat,
                         untried=self._legal(next_state))
        node.children[strat] = child
        return child

    def _backprop(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def _best_by_mean(self, root: MCTSNode):
        best, best_v = None, -1e18
        for strat, child in root.children.items():
            if child.mean_value > best_v:
                best, best_v = strat, child.mean_value
        return best if best is not None else self.strategies[0]