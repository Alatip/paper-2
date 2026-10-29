"""Error correlation: phi, mean phi, tetrachoric (cos-pi and exact bivariate normal; empty 2x2 cell -> Haldane +0.5), Kish n_eff.
Paper 1, section 2 (phi, phi-bar, n_eff, tetrachoric rho) and appendix C (empty-cell rule)."""
import itertools
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm


def phi(a, b):
    """phi = (p11 - pa pb) / sqrt(pa(1-pa) pb(1-pb)) for two 0/1 vectors (section 2). Same value on correctness or on
    error indicators. None when either margin is 0 or 1 (convention of the original analysis)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    pa, pb = a.mean(), b.mean()
    den = pa * (1 - pa) * pb * (1 - pb)
    if den == 0:
        return None
    return float(((a * b).mean() - pa * pb) / math.sqrt(den))


def mean_phi(rows):
    """phi-bar: mean of phi over all pairs (itertools.combinations order), undefined pairs (None) dropped; nan if none."""
    v = [phi(a, b) for a, b in itertools.combinations(list(rows), 2)]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def table2x2(a, b):
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


TETRA_RULES = ("haldane", "drop", "limit")
TETRA_RULE = "haldane"   # default empty-cell rule; operators set it from --tetra-rule


def _tetra_cells(a, b, rule):
    """2x2 cells (n11, n10, n01, n00) for the tetrachoric, or a str 'drop'/'+1'/'-1' / None.
    None: a vector is constant (undefined under every rule). Empty cell (some cell is 0), by `rule`:
    haldane = add 0.5 to EVERY cell (Haldane correction) and estimate as usual; drop = None (pair dropped from means); limit = the +-1 limit (+1 if n10 n01 = 0, else -1; sensitivity only)."""
    rule = rule or TETRA_RULE
    if rule not in TETRA_RULES:
        raise ValueError(f"tetrachoric rule must be one of {TETRA_RULES}, got {rule!r}")
    n11, n10, n01, n00 = table2x2(a, b)
    if min(n11 + n10, n01 + n00, n11 + n01, n10 + n00) == 0:
        return None
    if min(n11, n10, n01, n00) > 0:
        return float(n11), float(n10), float(n01), float(n00)
    if rule == "drop":
        return None
    if rule == "limit":
        return 1.0 if n10 * n01 == 0 else -1.0
    return n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5


def tetra_cospi(a, b, rule=None):
    """Tetrachoric rho, Pearson cos-pi approximation r = cos(pi / (1 + sqrt(n11 n00 / (n10 n01)))) (section 2;
    stats.tetra). None if a vector is constant. Empty cell: rule (default TETRA_RULE = 'haldane': +0.5 to every cell;
    'drop': None; 'limit': +-1), see _tetra_cells."""
    c = _tetra_cells(a, b, rule)
    if c is None or isinstance(c, float):
        return c
    n11, n10, n01, n00 = c
    return math.cos(math.pi / (1 + math.sqrt(n11 * n00 / (n10 * n01))))


def tetra_exact(a, b, rule=None):
    """Exact (ML on the 2x2 table, thresholds from margins) tetrachoric rho: solve
    Phi2(Phi^-1(pa), Phi^-1(pb); rho) = p11. Not used by the operators (they use cos-pi); provided for comparison.
    None if a vector is constant. Empty cell: same rule as tetra_cospi (haldane: margins and p11 from the +0.5 cells)."""
    c = _tetra_cells(a, b, rule)
    if c is None or isinstance(c, float):
        return c
    n11, n10, n01, n00 = c
    n = n11 + n10 + n01 + n00
    pa, pb, p11 = (n11 + n10) / n, (n11 + n01) / n, n11 / n
    ha, hb = norm.ppf(pa), norm.ppf(pb)
    f = lambda r: multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([ha, hb]) - p11
    return float(brentq(f, -0.9999, 0.9999, xtol=1e-8))


def mean_tetra(rows, method="cospi", rule=None):
    """Mean pairwise tetrachoric over rows (members) (as in the original analysis); pairs with a constant row are dropped, pairs
    with an empty cell follow `rule` (default TETRA_RULE = 'haldane', kept); nan if no pair is defined."""
    if method not in ("cospi", "exact"):  # guard: an unknown method must not fall back to 'exact'
        raise ValueError(f"mean_tetra: method must be 'cospi' or 'exact', got {method!r}")
    fn = tetra_cospi if method == "cospi" else tetra_exact
    v = [fn(a, b, rule) for a, b in itertools.combinations(list(rows), 2)]
    v = [x for x in v if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else float("nan")


def empty_cell_pairs(rows):
    """Number of pairs (over non-constant rows) whose 2x2 table has an empty cell."""
    return sum(isinstance(_tetra_cells(a, b, "limit"), float) for a, b in itertools.combinations(list(rows), 2))


def n_eff(n, phibar):
    """Kish design effect: n_eff = n / (1 + (n-1) phi-bar) (section 2). None if phibar is None."""
    if phibar is None:
        return None
    # guard: the denominator must stay positive, i.e. phi-bar > -1/(n-1); above n for phi-bar < 0
    if math.isnan(phibar) or n < 1 or phibar > 1 or (n > 1 and phibar <= -1 / (n - 1)):
        raise ValueError(f"n_eff: need n >= 1 and -1/(n-1) < phibar <= 1, got n={n}, phibar={phibar}")
    return n / (1 + (n - 1) * phibar)


def pair_matrix(rows, fn=None):
    """k x k matrix of a pairwise statistic over the rows (members) of a 0/1 matrix: fn = phi (default) or tetra_cospi /
    tetra_exact; undefined pairs (None) and the diagonal are nan. Used to average a statistic over many subsets."""
    fn = fn or phi
    rows = [np.asarray(r) for r in rows]
    k = len(rows)
    R = np.full((k, k), np.nan)
    for a, b in itertools.combinations(range(k), 2):
        v = fn(rows[a], rows[b])
        R[a, b] = R[b, a] = np.nan if v is None else v
    return R


def subset_mean(R, subsets):
    """Mean of the pairwise values R[a, b] over the pairs of every subset (rows of `subsets`, member indices), undefined
    pairs dropped (nan-mean; nan if none). Equals mean_phi / mean_tetra of the subset's rows when R = pair_matrix(...)."""
    S = np.asarray(subsets)
    k = S.shape[1]
    vals = np.stack([R[S[:, i], S[:, j]] for i, j in itertools.combinations(range(k), 2)], axis=1)
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(vals)).sum(1)
        return np.where(cnt > 0, np.nansum(vals, 1) / np.maximum(cnt, 1), np.nan)
