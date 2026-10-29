"""Independent checks of lib/p1stats: hand values, simulations with a known truth, identities and outside references
(scipy). Each test is written so that it fails if the formula is wrong, not merely if the implementation changes.
Nothing here reads the frozen .out files."""
import itertools
import math
import random
from statistics import NormalDist

import numpy as np
import pytest
from scipy.stats import binom, false_discovery_control, hypergeom

from p1stats import factor as F, metrics as M, tests_stats as T, voting as V

_Nd = NormalDist()


# =====================================================================================================================
# 1. Held-out (split-half) comparisons
# =====================================================================================================================
class FixedShuffle:
    """Fake random.Random: shuffle() rearranges the list into a prescribed order, so A and B halves are known."""

    def __init__(self, order):
        self.order = list(order)

    def shuffle(self, lst):
        lst[:] = list(self.order)


# Items 0..3 are group G1, items 4..7 are group G2.
#   m1 solves all of G1 and nothing of G2 (overall 4/8).
#   m2 solves 3 of G1 and all of G2 (overall 7/8) -- the better model overall and on G2.
# With A = G1, B = G2 the model chosen on A is m1 (4 > 3), yet on B it scores 0.
SCORE = {"m1": {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0},
         "m2": {0: 1, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}}
IDS = list(range(8))
POOL = ["m1", "m2"]


def _sb(i, m):
    return SCORE[m][i]


def test_split_half_uses_shuffled_order_and_halves():
    A, B = M.split_half(IDS, FixedShuffle([7, 6, 5, 4, 3, 2, 1, 0]))
    assert A == [7, 6, 5, 4] and B == [3, 2, 1, 0]
    # odd length: A gets N//2, B gets the rest
    A, B = M.split_half([1, 2, 3, 4, 5], FixedShuffle([5, 4, 3, 2, 1]))
    assert (A, B) == ([5, 4], [3, 2, 1])
    # real rng: a partition of the ids
    A, B = M.split_half(IDS, random.Random(3))
    assert sorted(A + B) == IDS and len(A) == 4 and not set(A) & set(B)


def test_heldout_chooses_on_A_scores_on_B():
    rng = FixedShuffle([0, 1, 2, 3, 4, 5, 6, 7])          # A = G1, B = G2
    op = lambda i: 0.5                                     # operator at a constant 0.5
    mean, lo, hi = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=rng)
    # correct: m1 chosen on A, scored on B -> 0 - 0.5
    assert mean == pytest.approx(-0.5) and lo == pytest.approx(-0.5) and hi == pytest.approx(-0.5)
    # wrong alternatives all give +0.5: scoring m1 on A (1 - 0.5), choosing the in-sample best m2 (1 - 0.5),
    # choosing the worst on A (m2, 1 - 0.5). So the value distinguishes them.
    assert mean != pytest.approx(0.5)
    # swap halves: A = G2 chooses m2 (4 > 0); on B = G1 m2 scores 3/4
    rng = FixedShuffle([4, 5, 6, 7, 0, 1, 2, 3])
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=rng)
    assert mean == pytest.approx(0.75 - 0.5)


def test_heldout_tie_on_A_goes_to_first_in_pool_order():
    # A = {3, 0, 1, 4}: m1 has 1+1+1+0 = 3, m2 has 0+1+1+1 = 3 -> tie -> first in pool order.
    rng = FixedShuffle([3, 0, 1, 4, 2, 5, 6, 7])
    op = lambda i: 0.0
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=rng)
    assert mean == pytest.approx(1 / 4)                    # m1 on B = {2, 5, 6, 7}: 1/4
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, ["m2", "m1"], n_splits=1, rng=rng)
    assert mean == pytest.approx(4 / 4)                    # m2 on B = {2, 5, 6, 7}: misses only item 3 -> 4/4


def test_heldout_keep_restricts_B_after_choice_and_counts_dropped():
    rng = FixedShuffle([0, 1, 2, 3, 4, 5, 6, 7])          # m1 chosen, B = {4,5,6,7}
    answered = {"m1": {4, 5}, "m2": {4, 5, 6, 7}}
    keep = lambda i, bm: i in answered[bm]
    dropped = []
    op = lambda i: 1.0 if i in (4, 6) else 0.0             # op right on 4 and 6
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=rng, keep=keep, dropped=dropped)
    # kept B = {4, 5}: m1 scores 0, op scores (1 + 0)/2 -> -0.5 ; without keep it would be (0 - 2/4) = -0.5 too,
    # so use an op that differs on dropped items:
    assert dropped == [2]
    op2 = lambda i: 1.0 if i in (6, 7) else 0.0            # op right only on the dropped items
    mean2, _, _ = M.heldout_best_minus(_sb, op2, IDS, POOL, n_splits=1, rng=FixedShuffle(range(8)), keep=keep)
    assert mean2 == pytest.approx(0.0)                     # on kept {4, 5}: 0 - 0
    mean3, _, _ = M.heldout_best_minus(_sb, op2, IDS, POOL, n_splits=1, rng=FixedShuffle(range(8)))
    assert mean3 == pytest.approx(-0.5)                    # whole B: 0 - 2/4
    # the keep filter depends on the chosen member: with pool reversed and A = G2, m2 is chosen and keeps all of B
    # keep removing everything -> error, not a silent nan
    with pytest.raises(ValueError):
        M.heldout_best_minus(_sb, op2, IDS, POOL, n_splits=1, rng=FixedShuffle(range(8)), keep=lambda i, bm: False)


def test_heldout_op_uses_member():
    rng = FixedShuffle([0, 1, 2, 3, 4, 5, 6, 7])          # m1 chosen, B = G2
    def op(i, m):
        return 1.0 if m == "m1" else 0.0
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=rng, op_uses_member=True)
    assert mean == pytest.approx(0 - 1.0)
    # same op through a different choice (A = G2 -> m2) gives op = 0 on B
    mean, _, _ = M.heldout_best_minus(_sb, op, IDS, POOL, n_splits=1, rng=FixedShuffle([4, 5, 6, 7, 0, 1, 2, 3]),
                                      op_uses_member=True)
    assert mean == pytest.approx(0.75 - 0.0)


def test_heldout_opposite_strengths_matches_hypergeometric_expectation():
    """Two models with disjoint solved halves of 20 items. In sample best = 0.5 exactly. Held out, the member that wins
    on A (the one with more of its items in A) has fewer of its items in B, so E[held-out best] < 0.5 and equals
    E[min(k, 10-k)/10] over k ~ Hypergeom(20, 10, 10) (a tie k = 5 gives 0.5 either way)."""
    ids = list(range(20))
    score = {"m1": {i: int(i < 10) for i in ids}, "m2": {i: int(i >= 10) for i in ids}}
    sb = lambda i, m: score[m][i]
    mean, lo, hi = M.heldout_best_minus(sb, lambda i: 0.0, ids, ["m1", "m2"], n_splits=2000, seed=0)
    ks = np.arange(11)
    expected = float(sum(hypergeom.pmf(k, 20, 10, 10) * min(k, 10 - k) / 10 for k in ks))
    assert expected < 0.5
    assert mean == pytest.approx(expected, abs=0.015)
    assert lo <= mean <= hi and hi <= 0.5
    # in-sample best ignores the split and says 0.5
    C = np.array([[score["m1"][i] for i in ids], [score["m2"][i] for i in ids]])
    assert M.best(C) == 0.5 and mean < 0.5 - 0.05


def test_heldout_single_member_mean_equals_overall_accuracy():
    """With one member the held-out best is just its accuracy on B; averaged over random halves it is the overall accuracy."""
    ids = list(range(30))
    c = {i: int(i % 3 == 0) for i in ids}                  # 10/30
    mean, lo, hi = M.heldout_best_minus(lambda i, m: c[i], lambda i: 0.2, ids, ["only"], n_splits=2000, seed=1)
    assert mean == pytest.approx(10 / 30 - 0.2, abs=0.01)
    assert lo < mean < hi


def test_routing_split_half_two_layers_opposite_winners():
    """Layer a: m solves all, n none; layer b: the reverse. The route on B is perfect (1.0), oracle 1.0,
    and the overall best (tie on A -> m, first in pool) scores exactly 1/2 on B for any split. share = 1."""
    X = {"a": {i: {"m": 1, "n": 0} for i in range(4)}, "b": {i: {"m": 0, "n": 1} for i in range(4)}}
    r = M.routing_split_half(X, ["m", "n"], ["a", "b"], n_splits=50, seed=0)
    assert r["route"] == pytest.approx(1.0) and r["oracle"] == pytest.approx(1.0)
    assert r["best"] == pytest.approx(0.5) and r["diff"] == pytest.approx(0.5)
    assert r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5) and r["share"] == pytest.approx(1.0)
    # with n first in the pool the tie goes to n; same numbers
    r2 = M.routing_split_half(X, ["n", "m"], ["a", "b"], n_splits=50, seed=0)
    assert r2["best"] == pytest.approx(0.5) and r2["route"] == pytest.approx(1.0)


def test_routing_split_half_is_below_in_sample_when_winner_flips():
    """Inside one layer the two models solve disjoint halves: the per-layer winner on A loses on B, so the held-out
    route is below the in-sample route (which is 0.5 by the tie rule) and below the in-sample oracle (1.0)."""
    X = {"L": {i: {"m": int(i < 10), "n": int(i >= 10)} for i in range(20)}}
    _, bacc, route, rc, orc = M.routing_in_sample(X, ["m", "n"], ["L"])
    assert bacc == 0.5 and rc == 0.5 and orc == 1.0 and route == {"L": "m"}
    r = M.routing_split_half(X, ["m", "n"], ["L"], n_splits=2000, seed=0)
    expected = float(sum(hypergeom.pmf(k, 20, 10, 10) * min(k, 10 - k) / 10 for k in range(11)))
    assert r["route"] == pytest.approx(expected, abs=0.015)
    assert r["best"] == pytest.approx(expected, abs=0.015)     # same member is best and winner -> same value
    assert r["route"] < rc and r["oracle"] == pytest.approx(1.0)


def test_routing_split_half_ignoring_split_would_differ():
    """A layer where the in-sample winner differs from the held-out mean: m solves items 0..5 of 10, n solves 4..9.
    In sample both have 6/10 -> tie -> m, route = 0.6. Held out the chosen winner's B accuracy averages below 0.6."""
    X = {"L": {i: {"m": int(i <= 5), "n": int(i >= 4)} for i in range(10)}}
    _, _, _, rc, _ = M.routing_in_sample(X, ["m", "n"], ["L"])
    assert rc == 0.6
    r = M.routing_split_half(X, ["m", "n"], ["L"], n_splits=2000, seed=0)
    assert r["route"] < 0.6 - 0.02


# =====================================================================================================================
# 2. Copula: rho -> 1 continuity, rho = 0 independence, Monte-Carlo reference
# =====================================================================================================================
PS_CASES = [
    [0.9, 0.5, 0.2],
    [0.9, 0.8, 0.6, 0.5],
    [0.95, 0.7, 0.6, 0.4, 0.3],
    [0.9, 0.85, 0.7, 0.55, 0.4, 0.3],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
]


def _lower_middle(ps):
    """Independent definition: the (n/2 + 1)-th best for even n, the (n+1)/2-th best for odd n."""
    s = sorted(ps, reverse=True)
    n = len(s)
    return s[n // 2] if n % 2 == 0 else s[(n - 1) // 2]


@pytest.mark.parametrize("ps", PS_CASES)
def test_median_member_hand(ps):
    assert V.median_member(ps) == _lower_middle(ps)
    assert V.median_member(ps, even="lower") == _lower_middle(ps)
    if len(ps) % 2 == 0:
        s = sorted(ps, reverse=True)
        assert V.median_member(ps, even="mean") == pytest.approx((s[len(s) // 2 - 1] + s[len(s) // 2]) / 2)
        assert V.median_member(ps) < V.median_member(ps, even="mean")   # lower of the two, strictly (distinct ps)
    with pytest.raises(ValueError):
        V.median_member(ps, even="upper")


def test_median_member_hand_even_values():
    assert V.median_member([0.9, 0.8, 0.6, 0.5]) == 0.6
    assert V.median_member([0.5, 0.9]) == 0.5
    assert V.median_member([0.3, 0.9, 0.6]) == 0.6
    assert V.median_member([0.9, 0.8, 0.6, 0.5], even="mean") == pytest.approx(0.7)


@pytest.mark.parametrize("ps", PS_CASES)
def test_copula_majority_rho_to_one_approaches_lower_median(ps):
    """Strict majority at rho = 1: an item is solved by more than n/2 members iff the (floor(n/2)+1)-th best solves it.
    The copula value must converge to that member's p as rho -> 1, and the rho >= 0.999 branch must equal it exactly."""
    target = _lower_middle(ps)
    vals = [V.copula_majority(ps, rho, K=2000) for rho in (0.9, 0.97, 0.99, 0.997)]
    errs = [abs(v - target) for v in vals]
    assert errs[-1] < 0.02, (vals, target)
    assert errs == sorted(errs, reverse=True), (vals, target)            # monotone approach
    assert V.copula_majority(ps, 1.0) == target
    assert V.copula_majority(ps, 0.999) == target
    # the limit is NOT the upper middle member for even n (would be the wrong continuity)
    if len(ps) % 2 == 0:
        upper = sorted(ps, reverse=True)[len(ps) // 2 - 1]
        assert abs(vals[-1] - upper) > 0.03


@pytest.mark.parametrize("ps", PS_CASES)
def test_copula_majority_rho_zero_is_poisson_binomial(ps):
    n = len(ps)
    # independent brute force: enumerate all 2^n outcomes
    brute = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        if sum(bits) > n / 2:
            brute += math.prod(p if b else 1 - p for p, b in zip(ps, bits))
    assert V.condorcet_heterogeneous(ps) == pytest.approx(brute, abs=1e-12)
    assert V.copula_majority(ps, 0.0) == pytest.approx(brute, abs=1e-6)


def test_poisson_binomial_equal_p_is_binomial_and_homogeneous_condorcet():
    n, p = 7, 0.6
    pmf = V.poisson_binomial_pmf([p] * n)
    assert pmf == pytest.approx(binom.pmf(np.arange(n + 1), n, p).tolist(), abs=1e-12)
    hand = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(4, 8))
    assert V.condorcet_homogeneous(n, p) == pytest.approx(hand, abs=1e-12)
    assert V.condorcet_homogeneous(4, p) == pytest.approx(sum(math.comb(4, k) * p ** k * (1 - p) ** (4 - k) for k in (3, 4)))


def _mc_one_factor(ps, rho, n_draw, seed):
    """Monte-Carlo one-factor Gaussian copula: Z_i = sqrt(rho) D + sqrt(1-rho) e_i, member i correct iff Z_i < Phi^-1(p_i).
    Returns (majority share, all-wrong share)."""
    g = np.random.default_rng(seed)
    D = g.standard_normal((n_draw, 1)); e = g.standard_normal((n_draw, len(ps)))
    Z = math.sqrt(rho) * D + math.sqrt(1 - rho) * e
    thr = np.array([_Nd.inv_cdf(p) for p in ps])
    C = Z < thr
    k = C.sum(1)
    return float((k > len(ps) / 2).mean()), float((k == 0).mean())


@pytest.mark.parametrize("ps,rho", [([0.9, 0.5, 0.2], 0.5), ([0.9, 0.8, 0.6, 0.5], 0.3), ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], 0.6)])
def test_copula_against_monte_carlo(ps, rho):
    maj, aw = _mc_one_factor(ps, rho, 400_000, seed=7)
    assert V.copula_majority(ps, rho) == pytest.approx(maj, abs=0.004)
    assert V.copula_all_wrong(ps, rho) == pytest.approx(aw, abs=0.004)
    # marginals are preserved by the copula: E_D[p_i(D)] = p_i
    d, w = V._grid(400)
    for i, p in enumerate(ps):
        m = sum(wi * V.copula_conditional(ps, rho, di)[i] for di, wi in zip(d, w)) / sum(w)
        assert m == pytest.approx(p, abs=1e-6)


@pytest.mark.parametrize("ps", PS_CASES)
def test_copula_all_wrong_limits(ps):
    prod = math.prod(1 - p for p in ps)
    assert V.copula_all_wrong(ps, 0.0) == pytest.approx(prod, abs=1e-6)
    # rho -> 1: all fail iff the best fails -> 1 - max(p); monotone increase in rho (comonotone failures pile up)
    vals = [V.copula_all_wrong(ps, rho, K=2000) for rho in (0.0, 0.5, 0.9, 0.99)]
    assert vals == sorted(vals)
    assert abs(vals[-1] - (1 - max(ps))) < 0.02
    assert all(vals[0] <= v <= 1 - max(ps) + 1e-9 for v in vals)
    with pytest.raises(ValueError):
        V.copula_all_wrong(ps, 1.0)
    with pytest.raises(ValueError):
        V.copula_all_wrong([0.0, 0.5], 0.3)


# =====================================================================================================================
# 3. factor.py
# =====================================================================================================================
def test_mp_edge_hand():
    assert F.mp_edge(7, 700) == pytest.approx(1.21)
    assert F.mp_edge(4, 4) == pytest.approx(4.0)
    assert F.mp_edge(1, 100) == pytest.approx(1.21)
    # the MP law for the correlation matrix of n i.i.d. rows of length N: the top eigenvalue sits at about lambda+
    rng = np.random.default_rng(11)
    n, N = 10, 4000
    top = np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, N)))).max()
    assert abs(top - F.mp_edge(n, N)) < 0.03


def test_eigen_check_iid_bernoulli_has_nothing_above():
    rng = np.random.default_rng(5)
    X = (rng.random((7, 600)) < np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.45])[:, None]).astype(int)
    ev, edge, null_q, n_above = F.eigen_check(X, np.random.default_rng(1), n_perm=200)
    assert n_above == 0
    assert len(ev) == 7 and ev[0] >= ev[-1] and sum(ev) == pytest.approx(7.0)
    assert edge == pytest.approx(F.mp_edge(7, 600))
    assert null_q < 1.5                                        # permutation null for 7 x 600 lies near the MP edge


def test_eigen_check_planted_one_factor_finds_exactly_one():
    g = np.random.default_rng(2)
    n, N, rho = 7, 600, 0.6
    D = g.standard_normal((1, N)); e = g.standard_normal((n, N))
    thr = np.array([_Nd.inv_cdf(p) for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.5, 0.45]])[:, None]
    X = (math.sqrt(rho) * D + math.sqrt(1 - rho) * e < thr).astype(int)
    ev, edge, null_q, n_above = F.eigen_check(X, np.random.default_rng(1), n_perm=200)
    assert n_above == 1
    assert ev[0] > 2.5 and ev[1] < edge
    # constant rows are dropped before the eigenvalues
    Xc = np.vstack([X, np.ones((1, N), int)])
    ev2, *_ = F.eigen_check(Xc, np.random.default_rng(1), n_perm=5)
    assert len(ev2) == n


def _sim_2pl(a, b, theta, g):
    P = 1 / (1 + np.exp(-np.outer(theta, a) + (a * b)[None, :]))
    return (g.random(P.shape) < P).astype(float)


def test_fit_recovers_2pl_parameters():
    g = np.random.default_rng(42)
    a = np.array([0.7, 1.0, 1.3, 1.8, 0.9, 1.5, 1.1, 0.8, 1.6, 1.2])
    b = np.array([-1.5, -1.0, -0.5, 0.0, 0.3, 0.6, 1.0, 1.4, -0.8, 0.9])
    theta = g.standard_normal(4000)
    X = _sim_2pl(a, b, theta, g)
    f = F.fit(X)
    assert np.abs(f["b"] - b).max() < 0.2, f["b"]
    assert np.abs(f["a"] - a).max() < 0.3, f["a"]
    assert np.corrcoef(f["a"], a)[0, 1] > 0.9
    assert np.corrcoef(f["b"], b)[0, 1] > 0.99
    assert np.corrcoef(f["theta"], theta)[0, 1] > 0.8
    # the parameterisation is P = expit(a (theta - b)): b = -delta/alpha and P matches
    assert f["b"] == pytest.approx(-f["delta"] / f["alpha"])
    P_hand = 1 / (1 + np.exp(-(f["a"][None, :] * (f["theta"][:, None] - f["b"][None, :]))))
    assert np.abs(f["P"] - P_hand).max() < 1e-9
    # eap_theta with the fitted parameters reproduces theta (fit's theta comes from the last E-step, one M-step earlier)
    th2, _ = F.eap_theta(X, f["alpha"], f["delta"])
    assert np.abs(th2 - f["theta"]).max() < 1e-3
    # Rasch fixes a = 1
    fr = F.fit(X, rasch=True)
    assert np.all(fr["a"] == 1.0)
    assert np.corrcoef(fr["b"], b)[0, 1] > 0.98


def test_lz_is_standardised_under_the_model():
    g = np.random.default_rng(3)
    a = np.full(60, 1.2); b = np.linspace(-2, 2, 60); theta = g.standard_normal(3000)
    X = _sim_2pl(a, b, theta, g)
    P = 1 / (1 + np.exp(-np.outer(theta, a) + (a * b)[None, :]))
    z = F.lz(X, P)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.08


def test_q3_bias_hand():
    assert F.q3_bias(7) == pytest.approx(-1 / 6)
    assert F.q3_bias(2) == -1.0
    with pytest.raises(ValueError):
        F.q3_bias(1)


def test_residual_correlation_independent_members_has_q3_bias():
    """Members (rows) conditionally independent given a one-factor 2PL: the mean residual correlation between members
    should sit at Yen's bias -1/(n-1), excess about 0. Checked with the fitted model's own residuals."""
    g = np.random.default_rng(8)
    n, N = 8, 1500
    theta = np.linspace(-1.2, 1.2, n)
    a = g.uniform(0.7, 1.6, N); b = g.normal(0, 1, N)
    X = _sim_2pl(a, b, theta, g)
    X, _ = F.drop_constant_items(X)
    mr, excess, E = F.residual_correlation(X)
    assert E.shape == X.shape
    assert mr == pytest.approx(F.q3_bias(n), abs=0.04), (mr, F.q3_bias(n))
    assert abs(excess) < 0.04
    # identity: mean off-diagonal of corrcoef(E), computed independently
    R = np.corrcoef(E)
    assert mr == pytest.approx(R[~np.eye(n, dtype=bool)].mean())
    # planted residual dependence between members 0..3 beyond the factor: excess clearly positive
    Xd = X.copy()
    shared = g.random(X.shape[1]) < 0.5
    Xd[:4, shared] = 1.0; Xd[:4, ~shared] = 0.0
    mr2, excess2, _ = F.residual_correlation(Xd)
    assert excess2 > 0.05 and mr2 > mr


def test_drop_constant_items_hand():
    X = np.array([[1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 1, 0]])
    Y, keep = F.drop_constant_items(X)
    assert keep.tolist() == [False, False, True, True]
    assert Y.tolist() == [[1, 1], [0, 1], [1, 0]]


def _family_E(n_fam, per_fam, N, gamma, seed):
    """Residual-like matrix: members in a family share gamma * family signal per item plus own noise."""
    g = np.random.default_rng(seed)
    fam = [f"f{k}" for k in range(n_fam) for _ in range(per_fam)]
    S = g.standard_normal((n_fam, N))
    E = np.vstack([gamma * S[k] + g.standard_normal(N) for k in range(n_fam) for _ in range(per_fam)])
    return E, fam


def test_family_stats_planted_within_family_correlation():
    E, fam = _family_E(4, 2, 300, gamma=0.8, seed=1)
    r = F.family_stats(E, fam, np.random.default_rng(0), n_perm=400)
    assert r["n_families"] == 4
    assert r["T_within"] > 0.3 and abs(r["T_between"]) < 0.1
    assert r["T"] == pytest.approx(r["T_within"] - r["T_between"])
    assert r["p_T"] < 0.02 and r["p_ICC"] < 0.02
    assert r["ICC"] > 0.2
    # hand check of T from corrcoef
    R = np.corrcoef(E)
    within = [R[i, j] for i in range(8) for j in range(8) if i != j and fam[i] == fam[j]]
    between = [R[i, j] for i in range(8) for j in range(8) if fam[i] != fam[j]]
    assert r["T_within"] == pytest.approx(np.mean(within)) and r["T_between"] == pytest.approx(np.mean(between))


def test_family_stats_null_p_not_small_on_average():
    ps = []
    for seed in range(8):
        E, fam = _family_E(4, 2, 200, gamma=0.0, seed=100 + seed)
        r = F.family_stats(E, fam, np.random.default_rng(seed), n_perm=150)
        ps.append(r["p_T"])
        assert abs(r["ICC"]) < 0.15
    ps = np.array(ps)
    assert ps.mean() > 0.25 and (ps < 0.05).sum() <= 2 and ps.max() > 0.3
    assert ps.min() >= 1 / 151                                 # (k + 1)/(n_perm + 1), never 0


# =====================================================================================================================
# 4. Holm and Benjamini-Hochberg
# =====================================================================================================================
def _holm_ref(p):
    """Independent brute force: adj_i = max over k with p_(k) <= p_i (by rank) of min(1, (m - k + 1) p_(k))."""
    m = len(p); s = sorted(range(m), key=lambda i: p[i]); out = [None] * m
    for pos, i in enumerate(s):
        out[i] = max(min(1.0, (m - k) * p[s[k]]) for k in range(pos + 1))
    return out


def _bh_ref(p):
    """Independent brute force: adj_i = min over k with rank >= rank(i) of min(1, m p_(k) / (k + 1))."""
    m = len(p); s = sorted(range(m), key=lambda i: p[i]); out = [None] * m
    for pos, i in enumerate(s):
        out[i] = min(min(1.0, m * p[s[k]] / (k + 1)) for k in range(pos, m))
    return out


def test_holm_hand():
    assert T.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert T.holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])       # input order preserved
    assert T.holm([0.6, 0.7]) == [1.0, 1.0]                                       # capped
    assert T.holm([0.02, 0.02, 0.5]) == pytest.approx([0.06, 0.06, 0.5])         # ties
    assert T.holm([0.001, 0.2, 0.3, 0.04]) == pytest.approx([0.004, 0.4, 0.4, 0.12])
    assert T.holm([0.3]) == [0.3] and T.holm([]) == []
    # step-down enforces monotonicity: a tiny p after a large one inherits the running max
    assert T.holm([0.05, 0.03, 0.9]) == pytest.approx([0.1, 0.09, 0.9])


def test_bh_hand():
    assert T.benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert T.benjamini_hochberg([0.04, 0.01, 0.03]) == pytest.approx([0.04, 0.03, 0.04])
    assert T.benjamini_hochberg([0.6, 0.7]) == pytest.approx([0.7, 0.7])
    assert T.benjamini_hochberg([0.5, 0.9]) == pytest.approx([0.9, 0.9])          # 0.5*2/1 = 1 -> min with 0.9
    assert T.benjamini_hochberg([0.6, 0.9]) == pytest.approx([0.9, 0.9])          # cap at 1 before the min
    assert T.benjamini_hochberg([0.02, 0.02, 0.5]) == pytest.approx([0.03, 0.03, 0.5])
    assert T.benjamini_hochberg([0.001, 0.2, 0.3, 0.04]) == pytest.approx([0.004, 0.2 * 4 / 3, 0.3, 0.08])
    assert T.benjamini_hochberg([0.3]) == [0.3] and T.benjamini_hochberg([]) == []


@pytest.mark.parametrize("seed", range(6))
def test_holm_bh_against_brute_force_and_scipy(seed):
    g = np.random.default_rng(seed)
    m = int(g.integers(1, 12))
    p = g.random(m).tolist()
    if seed % 2:
        p[: m // 2] = [round(x, 1) for x in p[: m // 2]]                           # introduce ties
    h, bh = T.holm(p), T.benjamini_hochberg(p)
    assert h == pytest.approx(_holm_ref(p))
    assert bh == pytest.approx(_bh_ref(p))
    assert bh == pytest.approx(false_discovery_control(p, method="bh").tolist())
    # properties: adj >= p; BH <= Holm <= Bonferroni; monotone in p; capped at 1
    bonf = [min(1.0, m * x) for x in p]
    for x, a, b_, c in zip(p, h, bh, bonf):
        assert x - 1e-12 <= b_ <= a + 1e-12 <= c + 1e-12 and a <= 1.0 and b_ <= 1.0
    order = sorted(range(m), key=lambda i: p[i])
    for u, v in zip(order, order[1:]):
        assert h[u] <= h[v] + 1e-12 and bh[u] <= bh[v] + 1e-12
    # BH step-up: the largest p is unchanged (unless m p > ... no: m p_(m)/m = p_(m))
    assert bh[order[-1]] == pytest.approx(min(1.0, p[order[-1]]))
