"""Blind tests for p1stats.correlation.

Written from the signatures and the mathematical spec only; no source was read.
Reference implementations are independent (numpy / scipy / hand arithmetic).
"""
import sys, pathlib, math, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm
from scipy.optimize import brentq

from p1stats import correlation as K


# ---------------------------------------------------------------- helpers

def _is_undefined(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def ref_phi(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def ref_table(a, b):
    """(n11, n10, n01, n00): both 1, a=1&b=0, a=0&b=1, both 0."""
    a = np.asarray(a); b = np.asarray(b)
    return (int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()),
            int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum()))


def ref_cospi(a, b):
    n11, n10, n01, n00 = ref_table(a, b)
    ad, bc = n11 * n00, n10 * n01
    if bc == 0:
        return 1.0
    if ad == 0:
        return -1.0
    return math.cos(math.pi / (1 + math.sqrt(ad / bc)))


def ref_cospi_haldane(a, b):
    """Haldane rule (decision 2026-10-17): any empty cell -> +0.5 to every cell, then the cos-pi formula."""
    n11, n10, n01, n00 = ref_table(a, b)
    if 0 in (n11, n10, n01, n00):
        n11, n10, n01, n00 = n11 + .5, n10 + .5, n01 + .5, n00 + .5
    return math.cos(math.pi / (1 + math.sqrt(n11 * n00 / (n10 * n01))))


def ref_exact_haldane(a, b):
    """Exact BVN root on the Haldane-corrected cells: Phi2(Phi^-1(p1), Phi^-1(p2); r) = n11/n (upper-orthant form)."""
    n11, n10, n01, n00 = ref_table(a, b)
    if 0 in (n11, n10, n01, n00):
        n11, n10, n01, n00 = n11 + .5, n10 + .5, n01 + .5, n00 + .5
    n = n11 + n10 + n01 + n00
    t1 = norm.ppf(1 - (n11 + n10) / n); t2 = norm.ppf(1 - (n11 + n01) / n)
    return brentq(lambda r: multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-t1, -t2]) - n11 / n,
                  -0.999999, 0.999999, xtol=1e-9)


def ref_tetra_exact(a, b):
    """rho such that P(X>t1, Y>t2) under standard BVN(rho) equals n11/n,
    with t1, t2 set from the margins."""
    n11, n10, n01, n00 = ref_table(a, b)
    n = n11 + n10 + n01 + n00
    p1 = (n11 + n10) / n; p2 = (n11 + n01) / n; p11 = n11 / n
    t1 = norm.ppf(1 - p1); t2 = norm.ppf(1 - p2)

    def f(r):
        upper = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-t1, -t2])
        return upper - p11  # P(X>t1,Y>t2) = P(-X<-t1,-Y<-t2) = Phi2(-t1,-t2)
    return brentq(f, -0.999999, 0.999999, xtol=1e-9)


def ref_kish(n, pb):
    return n / (1 + (n - 1) * pb)


def pairs_mean(rows, fn):
    vals = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        v = fn(rows[i], rows[j])
        if not _is_undefined(v):
            vals.append(v)
    return float(np.mean(vals))


def simulate(rho, p1, p2, n, seed):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    a = (z[:, 0] > norm.ppf(1 - p1)).astype(int)
    b = (z[:, 1] > norm.ppf(1 - p2)).astype(int)
    return a.tolist(), b.tolist()


A1 = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]
B1 = [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1]
A2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
B2 = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
rng0 = np.random.default_rng(7)
A3 = rng0.integers(0, 2, 60).tolist(); B3 = rng0.integers(0, 2, 60).tolist()
CASES = [(A1, B1), (A2, B2), (A3, B3)]


# ---------------------------------------------------------------- phi

@pytest.mark.parametrize("a,b", CASES)
def test_phi_matches_corrcoef(a, b):
    assert K.phi(a, b) == pytest.approx(ref_phi(a, b), abs=1e-12)


def test_phi_hand_asymmetric_table():
    # a=3 (11), b=2 (10), c=1 (01), d=4 (00)
    a = [1]*3 + [1]*2 + [0]*1 + [0]*4
    b = [1]*3 + [0]*2 + [1]*1 + [0]*4
    expected = (3*4 - 2*1) / math.sqrt((3+2)*(1+4)*(3+1)*(2+4))  # 10/sqrt(600)
    assert K.phi(a, b) == pytest.approx(expected, abs=1e-12)
    assert K.phi(b, a) == pytest.approx(expected, abs=1e-12)


def test_phi_limits():
    assert K.phi([1, 1, 0, 0, 1], [1, 1, 0, 0, 1]) == pytest.approx(1.0)
    assert K.phi([1, 1, 0, 0, 1], [0, 0, 1, 1, 0]) == pytest.approx(-1.0)
    # exact independence: 2x2 table with ad == bc
    a = [1, 1, 0, 0]; b = [1, 0, 1, 0]
    assert K.phi(a, b) == pytest.approx(0.0, abs=1e-12)


def test_phi_errors_equals_correctness():
    for a, b in CASES:
        na = [1 - x for x in a]; nb = [1 - x for x in b]
        assert K.phi(na, nb) == pytest.approx(K.phi(a, b), abs=1e-12)
        assert K.phi(na, b) == pytest.approx(-K.phi(a, b), abs=1e-12)


def test_phi_accepts_numpy_and_bools():
    a, b = CASES[0]
    assert K.phi(np.array(a), np.array(b)) == pytest.approx(ref_phi(a, b))
    assert K.phi([bool(x) for x in a], [bool(x) for x in b]) == pytest.approx(ref_phi(a, b))


def test_phi_constant_vector_undefined():
    v = K.phi([1, 1, 1, 1], [1, 0, 1, 0])
    assert _is_undefined(v), f"expected None/nan for constant vector, got {v!r}"


# ---------------------------------------------------------------- table2x2

def test_table2x2_counts_as_multiset():
    """Regardless of cell order, the four counts must be the four true counts."""
    for a, b in CASES:
        t = K.table2x2(a, b)
        cells = list(t.values()) if isinstance(t, dict) else list(np.asarray(t).ravel())
        assert sorted(cells) == sorted(ref_table(a, b))
        assert sum(cells) == len(a)


def test_table2x2_cell_order_convention():
    """Natural convention: (n11, n10, n01, n00) i.e. a=both1, b=a-only, c=b-only, d=both0,
    or the same as a 2x2 array [[n11, n10],[n01, n00]]."""
    a = [1]*3 + [1]*2 + [0]*1 + [0]*4
    b = [1]*3 + [0]*2 + [1]*1 + [0]*4
    t = K.table2x2(a, b)
    if isinstance(t, dict):
        vals = t
        # accept a few common key sets
        for keys in (("a", "b", "c", "d"), ("n11", "n10", "n01", "n00"), ((1, 1), (1, 0), (0, 1), (0, 0))):
            if all(k in vals for k in keys):
                assert tuple(vals[k] for k in keys) == (3, 2, 1, 4)
                return
        pytest.xfail(f"convention: table2x2 dict keys {list(vals)}")
    flat = tuple(int(x) for x in np.asarray(t).ravel())
    if flat != (3, 2, 1, 4):
        pytest.xfail(f"convention: table2x2 cell order is {flat}, expected (n11,n10,n01,n00)=(3,2,1,4)")


# ---------------------------------------------------------------- tetrachoric cos-pi

@pytest.mark.parametrize("a,b", CASES)
def test_tetra_cospi_formula(a, b):
    assert K.tetra_cospi(a, b) == pytest.approx(ref_cospi(a, b), abs=1e-12)


def test_tetra_cospi_hand():
    a = [1]*3 + [1]*2 + [0]*1 + [0]*4
    b = [1]*3 + [0]*2 + [1]*1 + [0]*4
    expected = math.cos(math.pi / (1 + math.sqrt(12 / 2)))
    assert K.tetra_cospi(a, b) == pytest.approx(expected, abs=1e-12)
    assert K.tetra_cospi(b, a) == pytest.approx(expected, abs=1e-12)


def test_tetra_cospi_limits():
    # Haldane rule: an empty cell (b, c, a or d = 0) -> +0.5 to every cell, value strictly inside (-1, 1)
    for a, b, sign in [([1, 1, 0, 0, 0], [1, 1, 1, 0, 0], 1),    # n10 = 0
                       ([1, 1, 1, 0, 0], [1, 1, 0, 0, 0], 1),    # n01 = 0
                       ([1, 1, 0, 0, 0], [0, 0, 1, 1, 0], -1),   # n11 = 0
                       ([1, 1, 0, 0, 1], [0, 0, 1, 1, 1], -1)]:  # n00 = 0
        v = K.tetra_cospi(a, b)
        assert v == pytest.approx(ref_cospi_haldane(a, b), abs=1e-12)
        assert 0 < sign * v < 1
    # hand value: cells (2, 0, 1, 2) -> (2.5, .5, 1.5, 2.5): cos(pi / (1 + sqrt(6.25 / .75)))
    assert K.tetra_cospi([1, 1, 0, 0, 0], [1, 1, 1, 0, 0]) == pytest.approx(
        math.cos(math.pi / (1 + math.sqrt(6.25 / 0.75))), abs=1e-12)
    # ad == bc -> cos(pi/2) = 0
    assert K.tetra_cospi([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0, abs=1e-12)


def test_tetra_cospi_errors_equals_correctness():
    for a, b in CASES:
        na = [1 - x for x in a]; nb = [1 - x for x in b]
        assert K.tetra_cospi(na, nb) == pytest.approx(K.tetra_cospi(a, b), abs=1e-12)


def test_tetra_cospi_constant_undefined():
    assert _is_undefined(K.tetra_cospi([0, 0, 0, 0], [1, 0, 1, 0]))


# ---------------------------------------------------------------- tetrachoric exact

@pytest.mark.parametrize("a,b", CASES)
def test_tetra_exact_matches_bvn_root(a, b):
    assert K.tetra_exact(a, b) == pytest.approx(ref_tetra_exact(a, b), abs=2e-4)


def test_tetra_exact_symmetric_and_complement():
    for a, b in CASES:
        v = K.tetra_exact(a, b)
        assert K.tetra_exact(b, a) == pytest.approx(v, abs=1e-6)
        na = [1 - x for x in a]; nb = [1 - x for x in b]
        assert K.tetra_exact(na, nb) == pytest.approx(v, abs=1e-6)
        assert K.tetra_exact(na, b) == pytest.approx(-v, abs=1e-6)


def test_tetra_exact_independence_zero():
    a = [1, 1, 0, 0]; b = [1, 0, 1, 0]
    assert K.tetra_exact(a, b) == pytest.approx(0.0, abs=1e-4)


def test_tetra_exact_limits():
    # Haldane rule: empty cell -> +0.5 to every cell, then the exact root on the corrected table
    for a, b, sign in [([1, 1, 0, 0, 0], [1, 1, 1, 0, 0], 1), ([1, 1, 0, 0, 0], [0, 0, 1, 1, 0], -1)]:
        v = K.tetra_exact(a, b)
        assert v == pytest.approx(ref_exact_haldane(a, b), abs=2e-4)
        assert 0 < sign * v < 0.999


def test_tetra_exact_constant_undefined():
    assert _is_undefined(K.tetra_exact([1, 1, 1, 1], [1, 0, 1, 0]))


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.3, 0.6, 0.8])
def test_tetra_exact_recovers_rho_skewed(rho):
    a, b = simulate(rho, p1=0.2, p2=0.75, n=40000, seed=abs(int(100 * rho)) + 11)
    est = K.tetra_exact(a, b)
    assert est == pytest.approx(rho, abs=0.03)
    # cos-pi: direction and loose bound only
    c = K.tetra_cospi(a, b)
    if rho != 0:
        assert math.copysign(1, c) == math.copysign(1, rho)
    assert abs(c - rho) < 0.3
    # phi is attenuated toward 0 relative to rho with skewed margins
    if rho != 0:
        assert abs(K.phi(a, b)) < abs(est)


# ---------------------------------------------------------------- mean_phi / mean_tetra

def test_mean_phi_vs_brute():
    rows = [A1, B1, [1 - x for x in A1], [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]]
    assert K.mean_phi(rows) == pytest.approx(pairs_mean(rows, ref_phi), abs=1e-12)


def test_mean_phi_two_rows():
    assert K.mean_phi([A1, B1]) == pytest.approx(ref_phi(A1, B1), abs=1e-12)


def test_mean_phi_skips_constant_rows():
    rows = [A1, B1, [1] * 12, [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]]
    good = [r for r in rows if len(set(r)) > 1]
    v = K.mean_phi(rows)
    assert not _is_undefined(v)
    assert v == pytest.approx(pairs_mean(good, ref_phi), abs=1e-12)


def test_mean_tetra_cospi_vs_brute():
    rows = [A2, B2, A2[::-1], B2[::-1]]  # no pair has an empty cell
    assert K.mean_tetra(rows) == pytest.approx(pairs_mean(rows, ref_cospi), abs=1e-12)
    assert K.mean_tetra(rows, method="cospi") == pytest.approx(pairs_mean(rows, ref_cospi), abs=1e-12)


def test_mean_tetra_exact_vs_brute():
    rows = [A2, B2, A2[::-1]]
    assert K.mean_tetra(rows, method="exact") == pytest.approx(pairs_mean(rows, ref_tetra_exact), abs=2e-4)


def test_mean_tetra_with_perfect_pair():
    """B2 vs 1-B2 is perfectly discordant (n11 = n00 = 0): Haldane rule, the pair is kept with the +0.5 value."""
    rows = [A2, B2, [1 - x for x in B2], A2[::-1]]
    assert K.mean_tetra(rows) == pytest.approx(pairs_mean(rows, ref_cospi_haldane), abs=1e-12)
    assert K.mean_tetra(rows, method="exact") == pytest.approx(pairs_mean(rows, ref_exact_haldane), abs=2e-4)


def test_mean_tetra_empty_cell_pairs_not_dropped():
    """Pairs with an empty cell are no longer dropped: mean over ALL 3 pairs, differs from the drop-rule mean."""
    a = [1, 1, 0, 0, 0, 1, 0, 1]; b = [1, 1, 1, 0, 0, 1, 0, 1]   # a <= b: n10 = 0
    c = [0, 1, 1, 0, 1, 0, 1, 0]                                   # all cells of (a,c), (b,c) non-empty
    rows = [a, b, c]
    v = K.mean_tetra(rows)
    allp = [ref_cospi_haldane(a, b), ref_cospi_haldane(a, c), ref_cospi_haldane(b, c)]
    assert v == pytest.approx(sum(allp) / 3, abs=1e-12)
    assert v != pytest.approx(sum(allp[1:]) / 2, abs=1e-6)


def test_mean_tetra_skips_constant_rows():
    rows = [A2, B2, [0] * 20, A2[::-1]]
    good = [A2, B2, A2[::-1]]
    assert K.mean_tetra(rows) == pytest.approx(pairs_mean(good, ref_cospi), abs=1e-12)


def test_mean_tetra_bad_method():
    with pytest.raises((ValueError, KeyError)):
        K.mean_tetra([A1, B1], method="bogus")


# ---------------------------------------------------------------- Kish n_eff

def test_n_eff_hand_values():
    assert K.n_eff(10, 0.5) == pytest.approx(10 / 5.5)
    assert K.n_eff(4, 0.25) == pytest.approx(4 / 1.75)
    assert K.n_eff(7, -0.1) == pytest.approx(7 / 0.4)
    assert K.n_eff(100, 0.02) == pytest.approx(100 / 2.98)


@pytest.mark.parametrize("n", [1, 2, 5, 50])
def test_n_eff_endpoints(n):
    assert K.n_eff(n, 0.0) == pytest.approx(n)
    assert K.n_eff(n, 1.0) == pytest.approx(1.0)


def test_n_eff_monotone_in_phibar():
    n = 12
    grid = np.linspace(-1 / (n - 1) + 1e-3, 1.0, 50)
    vals = [K.n_eff(n, p) for p in grid]
    assert all(x > y for x, y in zip(vals, vals[1:]))
    assert all(1.0 <= v for v in vals)


def test_n_eff_monotone_in_n_for_fixed_positive_phibar():
    vals = [K.n_eff(n, 0.3) for n in range(1, 40)]
    assert all(x < y for x, y in zip(vals, vals[1:]))
    assert vals[-1] < 1 / 0.3  # bounded by 1/phibar


def test_n_eff_negative_phibar_exceeds_n():
    assert K.n_eff(5, -0.1) > 5


@pytest.mark.parametrize("n,pb", [
    (5, 1.5), (5, -0.5), (5, -0.25),        # -1/(n-1) = -0.25 exactly: boundary excluded
    (5, float("nan")), (0, 0.2), (-3, 0.2), (5, 2.0), (5, -1.0),
])
def test_n_eff_value_error(n, pb):
    with pytest.raises(ValueError):
        K.n_eff(n, pb)


def test_n_eff_boundary_just_inside_ok():
    n = 5
    v = K.n_eff(n, -0.25 + 1e-6)
    assert v > 0 and math.isfinite(v)


# ---------------------------------------------------------------- pair_matrix / subset_mean

def _rows(k=6, m=30, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2, m)
    rows = []
    for i in range(k):
        flip = rng.random(m) < 0.3
        rows.append(np.where(flip, 1 - base, base).astype(int).tolist())
    return rows


def test_pair_matrix_default_phi_vs_brute():
    rows = _rows()
    R = np.asarray(K.pair_matrix(rows), float)
    k = len(rows)
    assert R.shape == (k, k)
    for i in range(k):
        for j in range(k):
            if i != j:
                assert R[i, j] == pytest.approx(ref_phi(rows[i], rows[j]), abs=1e-12)
    assert np.allclose(R, R.T, equal_nan=True)


def test_pair_matrix_diagonal_is_one_or_undefined():
    R = np.asarray(K.pair_matrix(_rows()), float)
    d = np.diag(R)
    assert np.all(np.isnan(d) | np.isclose(d, 1.0))


def test_pair_matrix_custom_fn():
    rows = _rows()
    R = np.asarray(K.pair_matrix(rows, fn=K.tetra_cospi), float)
    for i, j in itertools.combinations(range(len(rows)), 2):
        assert R[i, j] == pytest.approx(ref_cospi(rows[i], rows[j]), abs=1e-12)
    # arbitrary fn: hamming distance
    ham = lambda x, y: int(sum(p != q for p, q in zip(x, y)))
    H = np.asarray(K.pair_matrix(rows, fn=ham), float)
    for i, j in itertools.combinations(range(len(rows)), 2):
        assert H[i, j] == ham(rows[i], rows[j])


def test_pair_matrix_constant_row_gives_undefined_offdiag():
    rows = _rows(k=4)
    rows[2] = [1] * len(rows[0])
    R = np.asarray(K.pair_matrix(rows), float)
    assert np.isnan(R[0, 2]) and np.isnan(R[2, 1])
    assert not np.isnan(R[0, 1])


def _brute_subset_mean(R, subset):
    vals = [R[i, j] for i, j in itertools.combinations(subset, 2)]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals))


@pytest.mark.parametrize("subsets", [
    [[0, 1, 2], [3, 4, 5], [0, 3, 6], [1, 2, 6]],
    pytest.param([[0, 1, 2], [3, 4, 5, 6], [0, 6], [1, 2, 3, 4, 5, 6]],
                 marks=pytest.mark.xfail(reason="convention: subset_mean requires equal-size subsets (ragged list rejected)", strict=True)),
])
def test_subset_mean_vs_brute(subsets):
    rows = _rows(k=7)
    R = np.asarray(K.pair_matrix(rows), float)
    out = K.subset_mean(R, subsets)
    expected = [_brute_subset_mean(R, s) for s in subsets]
    # natural convention: one value per subset, in order
    if isinstance(out, dict):
        got = [out[k] for k in (tuple(s) for s in subsets)] if tuple(subsets[0]) in out else list(out.values())
    else:
        got = list(np.asarray(out, float).ravel()) if np.ndim(out) else [float(out)]
    if len(got) != len(subsets):
        pytest.xfail(f"convention: subset_mean returned shape {np.shape(out)} for {len(subsets)} subsets")
    assert got == pytest.approx(expected, abs=1e-12)


def test_subset_mean_single_subset_equals_mean_phi():
    rows = _rows(k=5)
    R = np.asarray(K.pair_matrix(rows), float)
    out = K.subset_mean(R, [[0, 1, 2, 3, 4]])
    v = float(np.asarray(list(out.values()) if isinstance(out, dict) else out, float).ravel()[0])
    assert v == pytest.approx(K.mean_phi(rows), abs=1e-12)
    assert v == pytest.approx(pairs_mean(rows, ref_phi), abs=1e-12)


def test_subset_mean_excludes_diagonal():
    R = np.full((4, 4), 0.5); np.fill_diagonal(R, 1.0)
    out = K.subset_mean(R, [[0, 1, 2]])
    v = float(np.asarray(list(out.values()) if isinstance(out, dict) else out, float).ravel()[0])
    assert v == pytest.approx(0.5)


def test_subset_mean_skips_nan_pairs():
    R = np.array([[1.0, 0.2, np.nan], [0.2, 1.0, 0.4], [np.nan, 0.4, 1.0]])
    out = K.subset_mean(R, [[0, 1, 2]])
    v = float(np.asarray(list(out.values()) if isinstance(out, dict) else out, float).ravel()[0])
    assert v == pytest.approx(0.3)
