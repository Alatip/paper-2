import math
import random

import numpy as np
import pytest
from scipy.stats import binom

from p1stats import correlation as K, metrics as M, tests_stats as T, voting as V
from p1stats.data import canon


# ---------- metrics
C = np.array([[1, 1, 0, 0, 1],
              [1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0]])


def test_metrics_hand():
    assert M.accuracy(C[0]) == 0.6
    assert M.oracle(C) == 0.8
    assert M.best(C) == 0.6 and M.best_index(C) == 0      # tie 0.6/0.6 -> first in pool order
    assert M.prize(C) == pytest.approx(0.2)
    assert M.all_wrong(C) == 1
    assert M.collected_share(0.7, 0.6, 0.8) == pytest.approx(0.5)
    assert math.isnan(M.collected_share(0.7, 0.8, 0.8))


def test_pct_convention():
    assert M.pct_2000(list(range(2000))) == (50, 1949)


def test_routing_in_sample_hand():
    X = {"a": {1: {"m": 1, "n": 0}, 2: {"m": 1, "n": 0}}, "b": {3: {"m": 0, "n": 1}, 4: {"m": 0, "n": 1}, 5: {"m": 1, "n": 1}}}
    b, bacc, route, r, o = M.routing_in_sample(X, ["m", "n"], ["a", "b"])
    assert b == "m" and bacc == 0.6 and route == {"a": "m", "b": "n"} and r == 1.0 and o == 1.0


def test_router_stochastic_hand():
    P = np.array([[1.0, 0.5], [0.0, 0.5]])
    best, rstar, orc = M.router_stochastic(P)
    assert best == 0.75 and rstar == 0.75 and orc == pytest.approx((1 + 0.75) / 2)


# ---------- correlation
def test_phi_hand():
    a = [1, 1, 0, 0]; b = [1, 0, 1, 0]
    assert K.phi(a, b) == 0
    assert K.phi(a, a) == pytest.approx(1)
    assert K.phi(a, [1 - x for x in a]) == pytest.approx(-1)
    assert K.phi(a, [1, 1, 1, 1]) is None
    # phi of errors equals phi of correctness
    rng = np.random.default_rng(1); x = rng.integers(0, 2, 50); y = rng.integers(0, 2, 50)
    assert K.phi(x, y) == pytest.approx(K.phi(1 - x, 1 - y))


def test_neff_paper_example():
    assert round(K.n_eff(7, 0.40), 2) == 2.06      # section 2 example
    assert round(K.n_eff(7, 0.68), 2) == 1.38
    assert K.n_eff(7, 0) == 7 and K.n_eff(7, 1) == 1


def test_tetra_cospi_hand():
    a = [1] * 30 + [1] * 10 + [0] * 10 + [0] * 30
    b = [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30
    assert K.tetra_cospi(a, b) == pytest.approx(math.cos(math.pi / (1 + 3)))
    assert K.tetra_cospi([1, 0, 1], [1, 0, 1], rule="drop") is None                  # old rule: empty cell -> undefined
    assert K.tetra_cospi([1, 0, 1], [1, 0, 1]) == pytest.approx(math.cos(math.pi / (1 + math.sqrt(2.5 * 1.5 / 0.25))))  # Haldane
    assert K.tetra_cospi([1, 1, 1], [1, 0, 1]) is None                               # constant row: undefined under every rule


def test_tetra_exact_recovers_rho():
    rng = np.random.default_rng(0); rho = 0.6
    Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 200000)
    a = Z[:, 0] > -0.3; b = Z[:, 1] > 0.5
    assert K.tetra_exact(a, b) == pytest.approx(rho, abs=0.01)
    assert K.tetra_cospi(a, b) == pytest.approx(rho, abs=0.06)   # cos-pi is biased upward with skewed margins (+0.05 here)


# ---------- voting
def test_plurality_ties():
    # 1-1 tie between A (wrong) and B (right): first in order wins
    assert V.plurality([("A", False), ("B", True)]) == 0.0
    assert V.plurality([("B", True), ("A", False)]) == 1.0
    assert V.plurality([("A", False), ("B", True)], rule="split") == 0.5
    assert V.plurality([(None, False), ("", False)]) == 0.0
    assert V.plurality([(None, False), ("24.00", True), ("$24", True), ("7", False)]) == 1.0   # canon merges numbers


def test_canon():
    assert canon("24") == canon("24.00") == canon("$24") and canon("5,072,530") == canon("5072530")
    assert canon("\\dfrac{1}{2}") == canon("1/2") == canon("0.5")


def test_majority_correct_even_tie_not_solved():
    assert V.majority_correct(np.array([[1, 1], [0, 1]])) == 0.5


def test_condorcet_homogeneous_is_binomial_tail():
    for n, p in [(3, 0.7), (7, 0.6), (4, 0.8)]:
        exp = sum(binom.pmf(k, n, p) for k in range(n + 1) if k > n / 2)
        assert V.condorcet_homogeneous(n, p) == pytest.approx(exp)
        assert V.condorcet_heterogeneous([p] * n) == pytest.approx(exp)
    assert V.condorcet_homogeneous(3, 0.7) == pytest.approx(0.784)


def test_poisson_binomial_hand():
    assert V.condorcet_heterogeneous([0.9, 0.8, 0.7]) == pytest.approx(0.9 * 0.8 * 0.7 + 0.9 * 0.8 * 0.3 + 0.9 * 0.2 * 0.7 + 0.1 * 0.8 * 0.7)


def test_copula_rho0_equals_poisson_binomial():
    ps = [0.9, 0.75, 0.6, 0.55, 0.8]
    assert V.copula_majority(ps, 0.0) == pytest.approx(V.condorcet_heterogeneous(ps), abs=1e-9)
    assert V.copula_all_wrong(ps, 0.0) == pytest.approx(np.prod([1 - p for p in ps]), abs=1e-12)


def test_copula_rho1_median_member():
    ps = [0.9, 0.75, 0.6, 0.55, 0.8]
    assert V.copula_majority(ps, 1.0) == 0.75
    assert V.copula_majority(ps, 0.98, K=4000) == pytest.approx(0.75, abs=0.02)   # continuity towards the median
    assert V.median_member([0.9, 0.8, 0.7, 0.6]) == 0.7 and V.median_member([0.9, 0.8, 0.7, 0.6], even="mean") == 0.75


def test_copula_preserves_marginals():
    # E_D p_i(D) = p_i
    d, w = V._grid(400)
    for p in (0.3, 0.9):
        m = sum(wi * V.copula_conditional([p], 0.5, di)[0] for di, wi in zip(d, w)) / sum(w)
        assert m == pytest.approx(p, abs=1e-4)


# ---------- tests_stats
def test_mcnemar():
    assert T.mcnemar_exact(0, 0) == 1.0
    assert T.mcnemar_exact(13, 4) == T.mcnemar_exact(4, 13)
    assert T.mcnemar_exact(10, 1) == pytest.approx(2 * 12 / 2 ** 11)
    assert T.mcnemar_exact(5, 5) == 1.0
    assert T.mcnemar_exact(2, 0) == 0.5
    assert T.mcnemar_exact("7", "0.0") == pytest.approx(2 / 2 ** 7)


P = [0.01, 0.04, 0.03, 0.005, 0.2]


def test_holm_bh_hand():
    # sorted: .005 .01 .03 .04 .2 ; Holm: .025 .04 .09 .09 .2 ; BH: .025 .025 .05 .05 .2
    assert T.holm(P) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2])
    assert T.benjamini_hochberg(P) == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.2])


def test_holm_bh_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(3); p = list(rng.random(30) ** 2)
    assert T.holm(p) == pytest.approx(list(sm.multipletests(p, method="holm")[1]))
    assert T.benjamini_hochberg(p) == pytest.approx(list(sm.multipletests(p, method="fdr_bh")[1]))


def test_wilson_known():
    lo, hi = T.wilson(1, 54)            # table 4.7: [0.3 %, 9.8 %]
    assert round(100 * lo, 1) == 0.3 and round(100 * hi, 1) == 9.8
    lo, hi = T.wilson(0, 10)
    assert lo == pytest.approx(0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = T.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert all(math.isnan(x) for x in T.wilson(0, 0))


def test_bootstrap_degenerate_and_seeded():
    assert T.paired_bootstrap_ci([0.5] * 20) == (0.5, 0.5, 0.5)
    a = T.paired_bootstrap_ci([0, 1, 0, 0, 1, -1], rng=random.Random(0))
    b = T.paired_bootstrap_ci([0, 1, 0, 0, 1, -1], seed=0)
    assert a == b and a[1] <= a[0] <= a[2]


# ---------- helpers added for operators/ (2026-10-16)
def test_plurality_key_raw_vs_canon():
    ans = [("24", True), ("24.0", True), ("7", False), ("7", False)]
    assert V.plurality(ans) == 1.0                      # canon: 24 == 24.0 -> 2 vs 2 tie, first is 24
    assert V.plurality(ans, key=lambda a: a) == 0.0     # raw: "7" has 2 votes, "24"/"24.0" one each


def test_heldout_keep_filter():
    ids = list(range(20)); pool = ["a", "b"]
    sb = lambda i, m: 1.0 if m == "a" else 0.0
    so = lambda i: 0.0
    m0 = M.heldout_best_minus(sb, so, ids, pool, n_splits=10)
    dr = []
    m1 = M.heldout_best_minus(sb, so, ids, pool, n_splits=10, keep=lambda i, bm: i % 2 == 0, dropped=dr)
    assert m0 == (1.0, 1.0, 1.0) and m1 == (1.0, 1.0, 1.0) and len(dr) == 10 and all(0 < d <= 10 for d in dr)


def test_heldout_op_uses_member():
    ids = list(range(10)); pool = ["a", "b"]
    sb = lambda i, m: 1.0 if m == "a" else 0.0
    assert M.heldout_best_minus(sb, lambda i, m: sb(i, m), ids, pool, n_splits=5, op_uses_member=True) == (0.0, 0.0, 0.0)


# guards added 2026-10-17 (Fable review, edge cases): these inputs must fail loudly, not return a number
import pytest as _pt
from p1stats import correlation as _c, voting as _v, tests_stats as _t, factor as _f


def test_guards():
    with _pt.raises(ValueError): _c.n_eff(7, -0.2)  # -0.2 < -1/6
    with _pt.raises(ValueError): _v.copula_majority([.9, .8, .7], -0.1)
    with _pt.raises(ValueError): _v.copula_all_wrong([.9, .8], 1.0)
    with _pt.raises(ValueError): _v.copula_all_wrong([1.0, .8], 0.3)
    with _pt.raises(ValueError): _t.paired_bootstrap_ci([])
    with _pt.raises(ValueError): _t.wilson(5, 3)
    with _pt.raises(ValueError): _f.q3_bias(1)


def test_copula_even_n_continuity():
    """Even n: the copula majority at rho close to 1 must approach the rho = 1 value (lower middle member)."""
    ps = [0.9, 0.8, 0.7, 0.6]
    assert abs(V.copula_majority(ps, 0.998) - V.copula_majority(ps, 1.0)) < 1e-3
    assert V.copula_majority(ps, 1.0) == 0.7
