"""Blind tests for operators/p2parts/setb.py, written from docs/SETB-SPEC.md and the signatures only."""
import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "operators"))
sys.path.insert(0, str(ROOT / "lib"))

from p2parts import answer_id as AID  # noqa: E402
from p2parts import setb as SB  # noqa: E402

AID.set_rule("p1")


def _ok(row):
    return row.get("ok", 0)


class TestConstants(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(SB.CAP, 64000)
        self.assertEqual(SB.NBOOT, 10000)
        self.assertEqual(SB.NSPLIT, 20)


class TestNormAnswer(unittest.TestCase):
    def test_spec_examples(self):
        self.assertEqual(SB.norm_answer("$\\dfrac{1}{5}$"), "\\frac{1}{5}")
        self.assertEqual(SB.norm_answer("\\arcsin\\!\\left(\\frac{1}{5}\\right)"), "\\arcsin(\\frac{1}{5})")
        self.assertEqual(SB.norm_answer("x = 2."), "x=2")

    def test_none_and_empty(self):
        self.assertEqual(SB.norm_answer(None), "")
        self.assertEqual(SB.norm_answer(""), "")

    def test_removals(self):
        self.assertEqual(SB.norm_answer("\\displaystyle 3\\,4\\;5\\( 6 \\)"), "3456")
        self.assertEqual(SB.norm_answer(" a\tb\nc "), "abc")
        self.assertEqual(SB.norm_answer("$$x$$"), "x")

    def test_one_trailing_dot_only(self):
        self.assertEqual(SB.norm_answer("x.."), "x.")
        self.assertEqual(SB.norm_answer("1.5"), "1.5")

    def test_trailing_dot_after_dollar_removed(self):
        # step 2 lists "$" removal before the trailing-dot removal: "$x = 1.$" -> "x=1." -> "x=1"
        self.assertEqual(SB.norm_answer("$x = 1.$"), "x=1")
        self.assertEqual(SB.norm_answer("2. "), "2")

    def test_frac_variants_and_overline(self):
        self.assertEqual(SB.norm_answer("\\tfrac{a}{b}"), "\\frac{a}{b}")
        self.assertEqual(SB.norm_answer("\\overline{R}"), "\\overlineR")

    def test_gamma_prefix(self):
        self.assertEqual(SB.norm_answer("\\Gamma(f) = 7"), "7")
        # only at the start
        self.assertEqual(SB.norm_answer("y+\\Gamma(f)=7"), "y+\\Gamma(f)=7")


class TestApplyAudit(unittest.TestCase):
    def setUp(self):
        self.items = {"a": {"gold": "1", "gold_norm": "1", "x": 0},
                      "b": {"gold": "2", "gold_norm": "2", "x": 0},
                      "c": {"gold": "3", "gold_norm": "3", "x": 0}}

    def test_relabel_and_grader_fix(self):
        audit = {"relabel": {"a": {"gold": "10", "gold_norm": "10", "why": "typo"}},
                 "grader_fix": {"b": {"gold": "20", "gold_norm": "20", "accept": ["$x = 1$", "\\dfrac{1}{2}"]},
                                "zzz": {"gold": "0", "gold_norm": "0", "accept": ["0"]}},
                 "other": {"c": {"gold": "99"}}}
        new, accept = SB.apply_audit(self.items, audit)
        self.assertEqual(new["a"], {"gold": "10", "gold_norm": "10", "x": 0})
        self.assertNotIn("why", new["a"])
        self.assertEqual(new["b"], {"gold": "20", "gold_norm": "20", "x": 0})
        self.assertNotIn("accept", new["b"])
        self.assertEqual(new["c"], self.items["c"])          # "other" key ignored
        self.assertNotIn("zzz", new)                          # ids absent from items ignored
        self.assertEqual(set(new), {"a", "b", "c"})
        self.assertEqual(accept["b"], {"x=1", "\\frac{1}{2}"})
        self.assertNotIn("a", accept)
        self.assertNotIn("c", accept)
        # Spec is silent on whether accept keeps ids absent from items ("zzz"); not asserted.

    def test_accept_normalised_with_trailing_dot_inside_dollars(self):
        # norm_answer: "$" removed everywhere, then one trailing "." removed (spec step 2 order)
        audit = {"grader_fix": {"a": {"gold": "1", "gold_norm": "1", "accept": ["$x = 1.$"]}}}
        _, accept = SB.apply_audit(self.items, audit)
        self.assertEqual(accept, {"a": {"x=1"}})

    def test_not_mutating_input(self):
        import copy
        before = copy.deepcopy(self.items)
        audit = {"relabel": {"a": {"gold": "10", "gold_norm": "10"}},
                 "grader_fix": {"a": {"gold": "11", "gold_norm": "11", "accept": ["11"]}}}
        new, accept = SB.apply_audit(self.items, audit)
        self.assertEqual(self.items, before)
        self.assertIsNot(new, self.items)
        self.assertIsNot(new["a"], self.items["a"])
        # grader_fix after relabel
        self.assertEqual(new["a"]["gold"], "11")
        self.assertEqual(accept, {"a": {"11"}})

    def test_empty_accept_absent(self):
        audit = {"grader_fix": {"a": {"gold": "1", "gold_norm": "1", "accept": []},
                                "b": {"gold": "2", "gold_norm": "2"}}}
        new, accept = SB.apply_audit(self.items, audit)
        self.assertEqual(accept, {})
        self.assertEqual(new, self.items)

    def test_empty_audit(self):
        new, accept = SB.apply_audit(self.items, {})
        self.assertEqual(new, self.items)
        self.assertEqual(accept, {})


class TestIsCorrect(unittest.TestCase):
    def test_falsy_rows(self):
        self.assertEqual(SB.is_correct(None, _ok), 0)
        self.assertEqual(SB.is_correct({}, lambda r: True), 0)

    def test_finish_and_cap(self):
        self.assertEqual(SB.is_correct({"finish": "length", "ok": 1}, _ok), 0)
        self.assertEqual(SB.is_correct({"finish": "error", "ok": 1}, _ok), 0)
        self.assertEqual(SB.is_correct({"finish": "stop", "ok": 1, "tok_out": 64001}, _ok), 0)
        self.assertEqual(SB.is_correct({"finish": "stop", "ok": 1, "tok_out": 64000}, _ok), 0)   # at the cap = cut (rule 2026-10-29)
        self.assertEqual(SB.is_correct({"finish": "stop", "ok": 1, "tok_out": 63999}, _ok), 1)
        self.assertEqual(SB.is_correct({"finish": "stop", "ok": 1, "tok_out": 11}, _ok, cap=10), 0)
        self.assertEqual(SB.is_correct({"finish": "stop", "ok": 1, "tok_out": None}, _ok), 1)

    def test_grade_fn_path(self):
        self.assertEqual(SB.is_correct({"ok": 1}, _ok), 1)
        self.assertEqual(SB.is_correct({"ok": 0}, _ok), 0)
        self.assertEqual(SB.is_correct({"ok": 1}, lambda r: "yes"), 1)
        self.assertIsInstance(SB.is_correct({"ok": 1}, lambda r: True), int)

    def test_accept_path_skips_grade_fn(self):
        calls = []

        def grade(r):
            calls.append(r)
            return True

        parse = lambda t: t.split("FINAL:")[-1]  # noqa: E731
        row = {"text": "blah FINAL: $x = 1$", "finish": "stop"}
        self.assertEqual(SB.is_correct(row, grade, accept={"x=1"}, parse_fn=parse), 1)
        self.assertEqual(SB.is_correct(row, grade, accept={"x=2"}, parse_fn=parse), 0)
        self.assertEqual(calls, [])
        # empty accept set -> never correct, text None handled as ""
        self.assertEqual(SB.is_correct({"text": None}, grade, accept=set(), parse_fn=lambda t: t), 0)
        self.assertEqual(calls, [])


class TestPlurality(unittest.TestCase):
    def test_simple_majority(self):
        c, top, keys = SB.plurality(["2", "3", "2"], [1, 0, 1])
        self.assertEqual(c, 1)
        self.assertEqual(top, keys[0])
        self.assertEqual(keys[0], keys[2])
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(len(keys), 3)

    def test_majority_wrong(self):
        c, top, keys = SB.plurality(["2", "3", "2"], [0, 1, 0])
        self.assertEqual(c, 0)
        self.assertEqual(top, keys[0])

    def test_equivalent_answers_grouped(self):
        c, top, keys = SB.plurality(["1/2", "3", "0.5"], [1, 0, 0])
        self.assertEqual(keys[0], keys[2])
        self.assertEqual(top, keys[0])
        self.assertEqual(c, 1)        # any member of winning group correct

    def test_tie_earliest_first_member_wins(self):
        c, top, keys = SB.plurality(["3", "2", "2", "3"], [0, 1, 1, 0])
        self.assertEqual(top, keys[0])
        self.assertEqual(c, 0)
        c, top, keys = SB.plurality(["2", "3"], [0, 1])
        self.assertEqual(top, keys[0])
        self.assertEqual(c, 0)

    def test_none_and_empty_string(self):
        c, top, keys = SB.plurality([None, "", "7"], [1, 1, 0])
        self.assertIsNone(keys[0])
        self.assertIsNone(keys[1])
        self.assertEqual(top, keys[2])
        self.assertEqual(c, 0)
        # None members do not form a winning group even if they are the majority
        c, top, keys = SB.plurality([None, None, "7"], [1, 1, 1])
        self.assertEqual(top, keys[2])
        self.assertEqual(c, 1)

    def test_no_answers(self):
        self.assertEqual(SB.plurality([], []), (0, None, []))
        c, top, keys = SB.plurality([None, ""], [1, 1])
        self.assertEqual((c, top), (0, None))
        self.assertEqual(keys, [None, None])

    def test_all_wrong(self):
        c, top, keys = SB.plurality(["2", "2", "2"], [0, 0, 0])
        self.assertEqual(c, 0)
        self.assertEqual(top, keys[0])


class TestUnanimous(unittest.TestCase):
    def test_cases(self):
        self.assertTrue(SB.is_unanimous_by_identity(["k", "k", "k"]))
        self.assertTrue(SB.is_unanimous_by_identity(["k"]))
        self.assertTrue(SB.is_unanimous_by_identity(["k", None, "k"]))
        self.assertFalse(SB.is_unanimous_by_identity(["k", "j"]))
        self.assertFalse(SB.is_unanimous_by_identity([None, None]))
        self.assertFalse(SB.is_unanimous_by_identity([]))


class TestNewCorrectAnswer(unittest.TestCase):
    def test_new_and_correct(self):
        self.assertEqual(SB.new_correct_answer("3", 1, ["2", "4"]), 1)
        self.assertEqual(SB.new_correct_answer("3", True, [None, "2"]), 1)
        self.assertEqual(SB.new_correct_answer("3", 1, []), 1)

    def test_not_new(self):
        self.assertEqual(SB.new_correct_answer("0.5", 1, ["1/2", "2"]), 0)
        self.assertEqual(SB.new_correct_answer("2", 1, ["2"]), 0)

    def test_not_ok_or_empty(self):
        self.assertEqual(SB.new_correct_answer("3", 0, ["2"]), 0)
        self.assertEqual(SB.new_correct_answer("", 1, ["2"]), 0)
        self.assertEqual(SB.new_correct_answer(None, 1, ["2"]), 0)


class TestRowCost(unittest.TestCase):
    PRICES = {"claude-x": (2.0, 10.0)}

    def test_falsy_row(self):
        self.assertEqual(SB.row_cost(None, "claude-x", self.PRICES), (0.0, False))
        self.assertEqual(SB.row_cost({}, "gpt", self.PRICES), (0.0, False))

    def test_non_claude(self):
        self.assertEqual(SB.row_cost({"cost": "0.25"}, "gpt", self.PRICES), (0.25, False))
        self.assertEqual(SB.row_cost({"cost": None, "tok_in": 5}, "gpt", {}), (0.0, False))
        self.assertEqual(SB.row_cost({"tok_in": 5}, "gpt", {}), (0.0, False))

    def test_claude_notional(self):
        self.assertEqual(SB.row_cost({"cost_notional": 0.5, "cost": 0}, "claude-x", self.PRICES), (0.5, False))
        self.assertEqual(SB.row_cost({"cost_notional": "0.5"}, "claude-x", self.PRICES), (0.5, False))
        self.assertEqual(SB.row_cost({"cost_notional": 0}, "claude-x", self.PRICES), (0.0, False))

    def test_claude_estimate(self):
        # pi_in*tok_in + pi_out*cap = 2*100 + 10*64000
        v, est = SB.row_cost({"tok_in": 100}, "claude-x", self.PRICES)
        self.assertTrue(est)
        self.assertAlmostEqual(v, 2.0 * 100 + 10.0 * 64000)
        v, est = SB.row_cost({"tok_in": 100, "cost_notional": ""}, "claude-x", self.PRICES, cap=10)
        self.assertTrue(est)
        self.assertAlmostEqual(v, 200 + 100)
        v, est = SB.row_cost({"cost_notional": None}, "claude-x", self.PRICES, cap=1)
        self.assertEqual((v, est), (10.0, True))


class TestSeedAndBootstrap(unittest.TestCase):
    def test_operator_seed(self):
        for name in ("debate7", "copies", ""):
            self.assertEqual(SB.operator_seed(name), int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
        self.assertNotEqual(SB.operator_seed("a"), SB.operator_seed("b"))

    def test_bootstrap_matches_reference(self):
        d = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=float)
        rng = np.random.default_rng(7)
        idx = rng.integers(0, len(d), size=(500, len(d)))
        means = d[idx].mean(axis=1)
        ref = [float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        got = SB.bootstrap_ci(d, np.random.default_rng(7), n_boot=500)
        self.assertEqual(got, ref)
        for x in got:
            self.assertIsInstance(x, float)

    def test_bootstrap_determinism_and_consumption(self):
        d = np.array([1, -1, 0, 1, 1, 0, -1, 1], dtype=float)
        a = SB.bootstrap_ci(d, np.random.default_rng(3), n_boot=200)
        b = SB.bootstrap_ci(d, np.random.default_rng(3), n_boot=200)
        self.assertEqual(a, b)
        rng = np.random.default_rng(3)
        c1 = SB.bootstrap_ci(d, rng, n_boot=200)
        c2 = SB.bootstrap_ci(d, rng, n_boot=200)
        self.assertEqual(c1, a)
        self.assertNotEqual(c1[1:], c2[1:])
        self.assertEqual(c1[0], c2[0])

    def test_bootstrap_constant(self):
        self.assertEqual(SB.bootstrap_ci(np.ones(5), np.random.default_rng(0), n_boot=50), [1.0, 1.0, 1.0])
        lo_hi = SB.bootstrap_ci(np.array([0.0, 1.0]), np.random.default_rng(0), n_boot=50)
        self.assertEqual(lo_hi[0], 0.5)
        self.assertLessEqual(lo_hi[1], lo_hi[0])
        self.assertGreaterEqual(lo_hi[2], lo_hi[0])


class TestSplits(unittest.TestCase):
    def test_shape_and_stratum_counts(self):
        strata = ["b"] * 5 + ["a"] * 4 + ["c"] * 1
        S = SB.splits(strata, n_split=6, seed=1)
        self.assertEqual(len(S), 6)
        for m in S:
            self.assertIsInstance(m, np.ndarray)
            self.assertEqual(m.dtype, np.bool_)
            self.assertEqual(m.shape, (10,))
            self.assertEqual(int(m[:5].sum()), 2)
            self.assertEqual(int(m[5:9].sum()), 2)
            self.assertEqual(int(m[9:].sum()), 0)

    def test_matches_reference_construction(self):
        strata = ["b", "a", "b", "a", "a", "b", "c", "c"]
        rng = np.random.default_rng(5)
        ref = []
        for _ in range(3):
            m = np.zeros(len(strata), dtype=bool)
            for lab in sorted(set(strata)):
                idx = np.array([i for i, s in enumerate(strata) if s == lab])
                rng.shuffle(idx)
                m[idx[: len(idx) // 2]] = True
            ref.append(m)
        got = SB.splits(strata, n_split=3, seed=5)
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g, r)

    def test_determinism_and_defaults(self):
        strata = list("aabbccddeeffgghh")
        S1 = SB.splits(strata)
        S2 = SB.splits(strata)
        self.assertEqual(len(S1), 20)
        for a, b in zip(S1, S2):
            np.testing.assert_array_equal(a, b)
        S3 = SB.splits(strata, n_split=20, seed=0)
        for a, b in zip(S1, S3):
            np.testing.assert_array_equal(a, b)


class TestHeldoutBest(unittest.TestCase):
    def test_hand_computed_two_participants(self):
        P = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
        m = np.array([True, True, False, False, False])
        # choose=m: col means (1, 0) -> j=0; score items 2,3,4 with col 0 -> 0,0,1
        # choose=~m: col means (1/3, 1) -> j=1; score items 0,1 with col 1 -> 0,0
        np.testing.assert_allclose(SB.heldout_best(P, [m]), [0, 0, 0, 0, 1])

    def test_two_masks_average(self):
        P = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
        m1 = np.array([True, True, False, False, False])
        m2 = np.array([False, False, True, True, False])
        # m2: choose=m2 -> col means (0,1) -> j=1; score 0,1,4 -> 0,0,1
        #     choose=~m2 -> (1, 1/3) -> j=0; score 2,3 -> 0,0
        np.testing.assert_allclose(SB.heldout_best(P, [m1, m2]), [0, 0, 0, 0, 1])
        m3 = np.array([True, False, True, False, True])
        # choose=m3 (items 0,2,4): col means (2/3, 2/3) -> tie -> j=0; score 1,3 -> P[1,0]=1, P[3,0]=0
        # choose=~m3 (items 1,3): (1/2, 1/2) -> j=0; score 0,2,4 -> 1,0,1
        np.testing.assert_allclose(SB.heldout_best(P, [m1, m3]), [0.5, 0.5, 0, 0, 1])

    def test_one_column(self):
        P = np.array([[1], [0], [1], [1]])
        m = np.array([True, False, True, False])
        np.testing.assert_allclose(SB.heldout_best(P, [m]), [1, 0, 1, 1])


class TestHeldoutBestUnderCost(unittest.TestCase):
    def setUp(self):
        self.acc = {"x": np.array([1, 1, 0, 0]), "y": np.array([0, 1, 1, 1]), "z": np.array([1, 1, 1, 1])}
        self.cost = {"x": np.array([1.0, 1, 1, 1]), "y": np.array([2.0, 2, 2, 2]), "z": np.array([9.0, 9, 9, 9])}
        self.m = np.array([True, True, False, False])

    def test_pick_best_affordable(self):
        CA = np.array([2.0, 2, 2, 2])
        # choose=m (0,1): x acc 1 cost 1; y acc .5; z unaffordable -> x; score 2,3 -> 0,0
        # choose=~m (2,3): x acc 0; y acc 1 -> y; score 0,1 -> 0,1
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, CA, [self.m], ["x", "y", "z"])
        np.testing.assert_allclose(v, [0, 1, 0, 0])
        self.assertEqual(picks, Counter({"x": 1, "y": 1}))
        self.assertIsInstance(picks, Counter)

    def test_tie_break_cheaper_then_name(self):
        acc = {"p": np.array([1, 1, 0, 0]), "q": np.array([1, 1, 0, 0]), "r": np.array([1, 1, 0, 0])}
        cost = {"p": np.array([2.0] * 4), "q": np.array([1.0] * 4), "r": np.array([1.0] * 4)}
        CA = np.array([3.0] * 4)
        v, picks = SB.heldout_best_under_cost(acc, cost, CA, [self.m], ["p", "q", "r"])
        self.assertEqual(picks, Counter({"r": 2}))   # equal acc; cheaper beats p; "r" > "q" lexicographically
        np.testing.assert_allclose(v, [1, 1, 0, 0])

    def test_no_affordable_returns_none(self):
        CA = np.array([0.5] * 4)
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, CA, [self.m], ["x", "y", "z"])
        self.assertIsNone(v)
        self.assertEqual(picks, Counter())
        # affordable for the first direction only: picks so far are returned
        CA = np.array([1.0, 1.0, 0.5, 0.5])
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, CA, [self.m], ["x", "y", "z"])
        self.assertIsNone(v)
        self.assertEqual(picks, Counter({"x": 1}))

    def test_two_masks_divide(self):
        self.acc["x"] = np.array([1, 0, 0, 0])
        CA = np.array([9.0] * 4)
        # choose=m (0,1): x .5, y .5, z 1 -> z; score 2,3 -> 1,1
        # choose=~m (2,3): y 1 cost 2, z 1 cost 9 -> y (cheaper); score 0,1 -> 0,1
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, CA, [self.m], ["x", "y", "z"])
        np.testing.assert_allclose(v, [0, 1, 1, 1])
        self.assertEqual(picks, Counter({"z": 1, "y": 1}))
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, CA, [self.m, ~self.m], ["x", "y", "z"])
        np.testing.assert_allclose(v, [0, 1, 1, 1])      # sum over 2 masks / 2
        self.assertEqual(picks, Counter({"z": 2, "y": 2}))


class TestResampleHeadroom(unittest.TestCase):
    def test_hand_computed(self):
        P = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0]])
        r = SB.resample_headroom(P, 0.5)
        self.assertAlmostEqual(r["mean_draw"], 5 / 12)
        self.assertAlmostEqual(r["pass_at_k"], 0.75)
        self.assertAlmostEqual(r["headroom"], 0.75 - 5 / 12)
        self.assertAlmostEqual(r["share"], (0.5 - 5 / 12) / (0.75 - 5 / 12))
        self.assertEqual(set(r), {"mean_draw", "pass_at_k", "headroom", "share"})

    def test_headroom_zero(self):
        r = SB.resample_headroom(np.array([[1, 1], [0, 0]]), 0.5)
        self.assertEqual(r["headroom"], 0)
        self.assertIsNone(r["share"])
        r = SB.resample_headroom(np.array([[1], [0], [1]]), 0.9)
        self.assertIsNone(r["share"])
        self.assertAlmostEqual(r["pass_at_k"], 2 / 3)


class TestOperatorStats(unittest.TestCase):
    def setUp(self):
        self.n = 12
        self.a = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0])
        self.a0 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        self.P = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 0],
                           [1, 1], [0, 0], [0, 0], [1, 0], [1, 0], [0, 1]])
        self.CA = np.full(self.n, 3.0)
        self.C0 = np.full(self.n, 1.0)
        self.acc = {"m0": self.P[:, 0], "m1": self.P[:, 1]}
        self.cost = {"m0": np.full(self.n, 1.0), "m1": np.full(self.n, 2.0)}
        self.strata = ["s1"] * 6 + ["s2"] * 6
        self.names = ["m0", "m1"]
        self.mc_calls = []

        def mcnemar(W, L):
            self.mc_calls.append((W, L))
            return 0.25

        self.mcnemar = mcnemar

    def run_stats(self, **kw):
        return SB.operator_stats("op", self.a, self.a0, self.P, self.CA, self.C0, self.acc, self.cost, self.strata,
                                 self.names, self.mcnemar, **kw)

    def test_basic_keys_and_values(self):
        out, rng = self.run_stats()
        self.assertEqual(out["N"], 12)
        self.assertAlmostEqual(out["acc"], 8 / 12)
        self.assertAlmostEqual(out["acc0"], 6 / 12)
        self.assertEqual(out["W"], 3)
        self.assertEqual(out["L"], 1)
        self.assertEqual(self.mc_calls, [(3, 1)])
        self.assertEqual(out["mcnemar_p"], 0.25)
        self.assertIsInstance(out["mcnemar_p"], float)
        self.assertAlmostEqual(out["C_usd"], 3.0)
        self.assertAlmostEqual(out["C0_usd"], 1.0)
        self.assertAlmostEqual(out["delta"][0], (out["W"] - out["L"]) / out["N"])
        self.assertAlmostEqual(out["usd_per_extra_item"], 2.0 / (2 / 12))
        self.assertNotIn("resample", out)
        self.assertIsInstance(rng, np.random.Generator)

    def test_rng_order_delta_then_gain(self):
        out, rng = self.run_stats()
        ref = np.random.default_rng(SB.operator_seed("op"))
        delta = SB.bootstrap_ci(self.a - self.a0, ref)
        self.assertEqual(out["delta"], delta)
        S = SB.splits(self.strata)
        v, picks = SB.heldout_best_under_cost(self.acc, self.cost, self.CA, S, self.names)
        self.assertIsNotNone(v)
        gain = SB.bootstrap_ci(self.a - v, ref)
        self.assertEqual(out["bestC_heldout"]["gain"], gain)
        self.assertAlmostEqual(out["bestC_heldout"]["acc"], float(v.mean()))
        self.assertEqual(out["bestC_heldout"]["picks"], dict(picks))
        self.assertIsInstance(out["bestC_heldout"]["picks"], dict)
        # the returned generator is the same advanced stream
        d = np.array([0.0, 1.0, 1.0])
        self.assertEqual(SB.bootstrap_ci(d, rng, 100), SB.bootstrap_ci(d, ref, 100))

    def test_determinism(self):
        o1, _ = self.run_stats()
        o2, _ = self.run_stats()
        self.assertEqual(o1, o2)

    def test_multi_column_keys(self):
        out, _ = self.run_stats()
        rowmax = self.P.max(axis=1)
        self.assertEqual(out["Cplus"], int(((rowmax == 0) & (self.a == 1)).sum()))
        self.assertEqual(out["Cplus"], 2)           # items 4 and 7
        self.assertAlmostEqual(out["oracle"], 8 / 12)
        self.assertAlmostEqual(out["best_insample"], 6 / 12)
        hb = float(SB.heldout_best(self.P, SB.splits(self.strata)).mean())
        self.assertAlmostEqual(out["best_heldout"], hb)
        self.assertAlmostEqual(out["prize"], out["oracle"] - out["best_heldout"])
        if out["oracle"] > out["best_heldout"]:
            self.assertAlmostEqual(out["s_heldout"], (out["acc"] - out["best_heldout"]) / out["prize"])
        else:
            self.assertIsNone(out["s_heldout"])

    def test_one_column_keys_absent(self):
        P1 = self.P[:, :1]
        acc = {"m0": P1[:, 0]}
        out, _ = SB.operator_stats("op", self.a, self.a0, P1, self.CA, self.C0, acc, {"m0": self.cost["m0"]},
                                   self.strata, ["m0"], self.mcnemar)
        for k in ("Cplus", "oracle", "best_heldout", "best_insample", "prize", "s_heldout"):
            self.assertNotIn(k, out)
        self.assertIn("bestC_heldout", out)
        self.assertEqual(out["bestC_heldout"]["picks"], {"m0": 40})

    def test_resample_flag(self):
        out, _ = self.run_stats(resample=True)
        self.assertEqual(out["resample"], SB.resample_headroom(self.P, float(self.a.mean())))

    def test_no_affordable_candidate(self):
        self.CA = np.full(self.n, 0.5)
        out, rng = self.run_stats()
        self.assertIsNone(out["bestC_heldout"])
        # only the delta CI consumed the rng
        ref = np.random.default_rng(SB.operator_seed("op"))
        SB.bootstrap_ci(self.a - self.a0, ref)
        d = np.array([0.0, 1.0, 1.0])
        self.assertEqual(SB.bootstrap_ci(d, rng, 100), SB.bootstrap_ci(d, ref, 100))

    def test_no_gain_gives_none_per_item(self):
        self.a = self.a0.copy()
        out, _ = self.run_stats()
        self.assertEqual(out["W"], 0)
        self.assertEqual(out["L"], 0)
        self.assertEqual(out["delta"], [0.0, 0.0, 0.0])
        self.assertIsNone(out["usd_per_extra_item"])
        self.a = np.zeros(self.n, dtype=int)
        out, _ = self.run_stats()
        self.assertLess(out["delta"][0], 0)
        self.assertIsNone(out["usd_per_extra_item"])
        self.assertEqual(out["acc"], 0.0)
        self.assertEqual(out["Cplus"], 0)

    def test_s_heldout_none_when_no_prize(self):
        P = np.ones((self.n, 2), dtype=int)
        acc = {"m0": P[:, 0], "m1": P[:, 1]}
        out, _ = SB.operator_stats("op", self.a, self.a0, P, self.CA, self.C0, acc, self.cost, self.strata,
                                   self.names, self.mcnemar)
        self.assertEqual(out["oracle"], 1.0)
        self.assertEqual(out["best_heldout"], 1.0)
        self.assertEqual(out["prize"], 0.0)
        self.assertIsNone(out["s_heldout"])
        self.assertEqual(out["Cplus"], 0)


class TestPluralityCodes(unittest.TestCase):
    def test_majority_correct(self):
        self.assertEqual(SB.plurality_codes(["a", "b", "a"], [1, 0, 1]), (1, "a"))

    def test_majority_wrong(self):
        self.assertEqual(SB.plurality_codes(["a", "b", "a"], [0, 1, 0]), (0, "a"))

    def test_tie_earliest_first_occurrence(self):
        self.assertEqual(SB.plurality_codes(["b", "a", "a", "b"], [0, 1, 1, 0]), (0, "b"))
        self.assertEqual(SB.plurality_codes(["a", "b"], [0, 1]), (0, "a"))

    def test_correct_iff_any_member_ok(self):
        self.assertEqual(SB.plurality_codes(["a", "a", "b"], [0, 1, 1]), (1, "a"))
        self.assertEqual(SB.plurality_codes(["a", "a", "b"], [0, 0, 1]), (0, "a"))

    def test_empty_and_none_ignored(self):
        self.assertEqual(SB.plurality_codes(["", None, "", "a"], [1, 1, 1, 0]), (0, "a"))
        self.assertEqual(SB.plurality_codes([None, "x", "", "y", "y"], [1, 1, 1, 0, 1]), (1, "y"))

    def test_all_empty(self):
        self.assertEqual(SB.plurality_codes(["", None, ""], [1, 1, 1]), (0, None))
        self.assertEqual(SB.plurality_codes([], []), (0, None))

    def test_single_answer(self):
        self.assertEqual(SB.plurality_codes(["z"], [1]), (1, "z"))
        self.assertEqual(SB.plurality_codes(["z"], [0]), (0, "z"))


class TestNewCorrectCode(unittest.TestCase):
    def test_new_and_correct(self):
        self.assertEqual(SB.new_correct_code("c", 1, ["a", "b"]), 1)
        self.assertEqual(SB.new_correct_code("c", True, ["a", "b"]), 1)

    def test_op_ok_falsy(self):
        self.assertEqual(SB.new_correct_code("c", 0, ["a", "b"]), 0)
        self.assertEqual(SB.new_correct_code("c", None, ["a", "b"]), 0)
        self.assertEqual(SB.new_correct_code("c", False, ["a", "b"]), 0)

    def test_empty_op_code(self):
        self.assertEqual(SB.new_correct_code("", 1, ["a"]), 0)
        self.assertEqual(SB.new_correct_code(None, 1, ["a"]), 0)

    def test_code_present_in_inputs(self):
        self.assertEqual(SB.new_correct_code("a", 1, ["a", "b"]), 0)
        self.assertEqual(SB.new_correct_code("a", 1, ["b", "", "a"]), 0)

    def test_empty_inputs(self):
        self.assertEqual(SB.new_correct_code("a", 1, []), 1)
        self.assertEqual(SB.new_correct_code("a", 0, []), 0)


if __name__ == "__main__":
    unittest.main()
