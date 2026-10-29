"""Significance tests and intervals: exact McNemar, paired bootstrap, Holm, Benjamini-Hochberg, Wilson.
Paper 1, section 4 (protocol: McNemar on discordant items, bootstrap 95% CI) and section 4.3 / appendix C (multiple comparisons)."""
import math
import random

from .metrics import pct_2000


def mcnemar_exact(b, c):
    """Exact two-sided McNemar from discordant counts: p = min(1, 2 sum_{j<=min(b,c)} C(n,j) / 2^n), n = b+c;
    n = 0 -> 1.0. Counts are int(float(.)) cast as in the original analysis."""
    fb, fc = float(b), float(c)
    if not (fb.is_integer() and fc.is_integer()) or fb < 0 or fc < 0:  # guard: counts like 9.5 are an upstream bug
        raise ValueError(f"mcnemar_exact: counts must be non-negative integers, got b={b}, c={c}")
    b, c = int(fb), int(fc)
    n = b + c; k = min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)


def paired_bootstrap_ci(d, B=2000, rng=None, seed=0):
    """Bootstrap over items of the mean paired difference d (panel - best): each resample draws len(d) indices by
    rng.randrange (random.Random), returns (mean(d), lo, hi) with the 2,000 percentile convention v[50], v[1949].
    Pass the shared rng to reproduce scripts that interleave several bootstraps on one stream."""
    if not d:
        raise ValueError("paired_bootstrap_ci: empty difference list")
    rng = rng or random.Random(seed)
    bs = []
    for _ in range(B):
        s = [d[rng.randrange(len(d))] for _ in d]
        bs.append(sum(s) / len(s))
    bs.sort()
    lo, hi = pct_2000(bs)
    return sum(d) / len(d), lo, hi


def holm(pvals):
    """Holm step-down adjusted p-values (section 4.3): sort ascending (stable), adj_(i) = max_{k<=i} min(1, (m-k) p_(k)),
    returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); out[i] = run
    return out


def benjamini_hochberg(pvals):
    """BH step-up adjusted p-values (section 4.3): adj_(i) = min_{k>=i} p_(k) m / k (cap 1), input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m; prev = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]; prev = min(prev, pvals[i] * m / (r + 1)); out[i] = prev
    return out


def wilson(k, n, z=1.96):
    """Wilson score interval (section 4.2 certification, table 4.7): (c -+ h)/d with d = 1 + z^2/n, c = p + z^2/(2n),
    h = z sqrt(p(1-p)/n + z^2/(4n^2)). n = 0 -> (nan, nan) (analyze.wilson; stats.cp_interval returns (0, 0))."""
    if n == 0:
        return (float("nan"), float("nan"))
    if n < 0 or not (0 <= k <= n):
        raise ValueError(f"wilson: need 0 <= k <= n, got k={k}, n={n}")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
