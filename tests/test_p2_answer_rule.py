"""Paper 2 uses paper 1's answer identity by default (decision 2026-10-18); the v8 rule stays behind P2_ANSWER_RULE=v8."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "operators"))
sys.path.insert(0, str(ROOT / "lib"))

from p1stats import voting as V  # noqa: E402
from p2parts import answer_id as AID  # noqa: E402
from p2parts import tn5  # noqa: E402
import debate_weak, debate_bounds  # noqa: E402


@pytest.fixture
def p1():
    prev = AID.RULE; AID.set_rule("p1"); yield; AID.set_rule(prev)


@pytest.fixture
def v8():
    prev = AID.RULE; AID.set_rule("v8"); yield; AID.set_rule(prev)


def test_default_is_paper1_rule():
    if "P2_ANSWER_RULE" not in os.environ:
        assert AID.RULE == "p1"


def test_keys_are_paper1_answer_groups(p1):
    xs = ["0.06", "0.1", "0.14", "1/2", "0.5", "\\frac{1}{2}", "50627678", "50627679", "5072530", "5072529.6", None, "B", "b"]
    assert AID.keys(xs) == V.answer_groups(xs)


def test_paper1_identity_cases(p1):
    k = AID.keys(["48179.42857", "48179.43", "48179"])
    assert k[0] == k[1] != k[2]                               # decimal rounding; integers never round-match
    k = AID.keys(["50627678", "50627679"]); assert k[0] != k[1]   # 12 digits / exact integers
    k = AID.keys(["0.06", "0.1", "0.14"]); assert k[0] == k[1] == k[2]   # chaining
    k = AID.keys(["\\frac{1}{2}", "0.5", "$1/2$"]); assert len(set(k)) == 1


def test_v8_rule_kept(v8):
    k = AID.keys(["50627678", "50627679"], tn5.canon); assert k[0] == k[1]   # 7 significant digits merge them
    k = AID.keys(["0.06", "0.1"], tn5.canon); assert k[0] != k[1]   # no decimal rounding in v8


def test_operators_majority_follow_rule(p1):
    rows = [dict(answer=a, correct=c) for a, c in (("5072530", True), ("5072529.6", False), ("5072529.6", False), ("5072530", True), ("5072530", True))]
    for mod in (tn5, debate_weak):
        assert mod.majority([r["answer"] for r in rows])[1] == 3
        assert mod.maj_correct(rows) is True
    assert debate_bounds.majority([r["answer"] for r in rows])[1] == 3
    # group correct if any member is (paper 1): 0.5 (wrong-graded) and 1/2 (right) are one answer and win 2:1
    rows2 = [dict(answer="0.5", correct=False), dict(answer="1/2", correct=True), dict(answer="7", correct=False)]
    assert tn5.maj_correct(rows2) is True and debate_weak.maj_correct(rows2) is True


def test_operators_v8_majority(v8):
    rows = [dict(answer="50627678", correct=True), dict(answer="50627679", correct=False)]
    assert tn5.majority([r["answer"] for r in rows])[1] == 2


