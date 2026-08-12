"""Unit tests for GBNF grammar construction (Phase 3)."""

from __future__ import annotations

import unittest

from src.agents.schemas import MacroStrategy, MacroStrategyDecision, TacticalCommand, TacticalKeyword
from src.inference.grammar import enum_choice_grammar, primary_enum_field


class TestGrammar(unittest.TestCase):
    def test_enum_choice_grammar_lists_all_values(self):
        g = enum_choice_grammar(["a", "b", "c"])
        self.assertIn("root ::=", g)
        for v in ("a", "b", "c"):
            self.assertIn(f'"{v}"', g)
        self.assertEqual(g.count("|"), 2)  # 3 alternatives -> 2 separators

    def test_enum_choice_grammar_rejects_empty(self):
        with self.assertRaises(ValueError):
            enum_choice_grammar([])

    def test_primary_enum_field_finds_keyword(self):
        name, cls = primary_enum_field(TacticalCommand)
        self.assertEqual(name, "keyword")
        self.assertIs(cls, TacticalKeyword)

    def test_primary_enum_field_finds_strategy(self):
        name, cls = primary_enum_field(MacroStrategyDecision)
        self.assertEqual(name, "strategy")
        self.assertIs(cls, MacroStrategy)

    def test_grammar_for_tactical_keyword_has_all_seven(self):
        _, cls = primary_enum_field(TacticalCommand)
        g = enum_choice_grammar([e.value for e in cls])
        self.assertEqual(len(list(cls)), 7)
        for e in cls:
            self.assertIn(f'"{e.value}"', g)


if __name__ == "__main__":
    unittest.main()