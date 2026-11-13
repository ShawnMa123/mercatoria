"""Fixtures stay internally consistent (P1-3 / P0-2)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "corpus" / "eval" / "test_cases.json"
HOLDOUT = ROOT / "tests" / "eval" / "holdout_cases.json"
TRADE = ROOT / "tests" / "eval" / "trade_law_questions.json"


class HoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dev = json.loads(DEV.read_text(encoding="utf-8"))
        self.holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))

    def test_holdout_ids_come_from_dev(self) -> None:
        dev_ids = {str(case["id"]) for case in self.dev}
        holdout_ids = {str(case["id"]) for case in self.holdout["cases"]}
        self.assertEqual(len(holdout_ids), self.holdout["_meta"]["size"])
        self.assertTrue(holdout_ids <= dev_ids)

    def test_dev_runs_exclude_holdout(self) -> None:
        dev_ids = {str(case["id"]) for case in self.dev}
        holdout_ids = {str(case["id"]) for case in self.holdout["cases"]}
        tuned = dev_ids - holdout_ids
        self.assertFalse(tuned & holdout_ids)
        self.assertEqual(len(tuned) + len(holdout_ids), len(dev_ids))

    def test_holdout_is_stratified(self) -> None:
        categories = {case["category"] for case in self.holdout["cases"]}
        self.assertGreaterEqual(len(categories), 20)


class TradeFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = json.loads(TRADE.read_text(encoding="utf-8"))["questions"]

    def test_every_gap_has_a_note(self) -> None:
        for case in self.questions:
            if case.get("corpus_gap"):
                self.assertTrue(
                    case.get("coverage_note"), f"{case['id']} is a gap without a note"
                )

    def test_no_answer_cases_are_not_gaps(self) -> None:
        for case in self.questions:
            if case.get("type") == "no_answer":
                self.assertFalse(case.get("corpus_gap"), case["id"])

    def test_no_answer_set_is_large_enough(self) -> None:
        typed = [case for case in self.questions if case.get("type") == "no_answer"]
        self.assertGreaterEqual(len(typed), 15)

    def test_ids_are_unique(self) -> None:
        ids = [case["id"] for case in self.questions]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
