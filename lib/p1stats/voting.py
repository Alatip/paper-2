"""Voting: plurality on answers, majority by correctness, Condorcet (homogeneous / Poisson-binomial), one-factor Gaussian
copula, median member. Paper 1, section 2 (plurality, tie rules) and section 3.4 (three predictions for majority)."""
import collections
import math
from statistics import NormalDist, median

import numpy as np
from scipy.stats import binom

from . import data as _data


_Nd = NormalDist()


def answer_groups(answers, same=None):
    """Group id per answer: the transitive closure (union-find) over one item's answers of
    data.same_answer = same canon() key OR the decimal-rounding rule (data.decimal_round_match: the longer decimal,
    rounded half-up to the places of the shorter, equals the shorter; integers exact; a/b and scientific literals by
    exact value). answers: list of answer strings (None/'' allowed). Returns a list of the same length: None for an
    empty answer, else the index of the first answer of its group (so ids do not depend on anything but the partition
    and the order given; the partition itself is order independent).
    Chaining caveat: the closure can join answers that are not directly related, e.g. 0.06 ~ 0.1 (0.06 -> 0.1) and
    0.14 ~ 0.1 (0.14 -> 0.1) put 0.06, 0.1 and 0.14 into one group although 0.06 and 0.14 differ at 2 places.
    same: pairwise relation (default data.same_answer, resolved at call time so tests can swap in same_answer_v1)."""
    same = same or _data.same_answer
    n = len(answers)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    live = [i for i, a in enumerate(answers) if a not in (None, "")]
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            if find(i) != find(j) and same(answers[i], answers[j]):
                ri, rj = find(i), find(j)
                parent[max(ri, rj)] = min(ri, rj)
    first = {}
    out = []
    for i, a in enumerate(answers):
        if a in (None, ""):
            out.append(None)
        else:
            out.append(first.setdefault(find(i), i))
    return out


def plurality(answers, rule="first", key=None):
    """Plurality of answers (section 2). answers: list of (answer, correct) in pool order; an empty/None
    answer is no vote (the member still counts wrong elsewhere). Returns score of the panel answer:
    rule='first' -> the tied answer whose first vote comes first in the given order wins (original voting analysis,
      original judge analysis; with order=strongest-first this is the 'strongest judge' rule of panel - best);
    rule='split' -> share of tied answers that are correct (tie counted fractionally).
    key: None (default) -> answer identity = answer_groups (canon key OR decimal-rounding rule, transitive closure); a function -> identity = key(answer) (key=lambda a: a gives the raw-string identity of
      the judge panels).
    No votes -> 0.0. Returns float."""
    answers = [(a, ok) for a, ok in answers if a not in (None, "")]
    if key is None:
        g = answer_groups([a for a, _ in answers])
        v = [(gi, bool(ok)) for gi, (_, ok) in zip(g, answers)]
    else:
        v = [(key(a), bool(ok)) for a, ok in answers]
    if not v:
        return 0.0
    c = collections.Counter(a for a, _ in v)
    t = max(c.values())
    right = {a for a, ok in v if ok}
    if rule == "split":
        tied = [a for a in c if c[a] == t]
        return sum(a in right for a in tied) / len(tied)
    first = next(a for a, _ in v if c[a] == t)
    return 1.0 if first in right else 0.0


def plurality_rows(d, pool, rule="first", key=None):
    """plurality() on {model: row} with row['answer'], row['correct'], in pool order (original voting analysis)."""
    return plurality([(d[m].get("answer"), d[m]["correct"]) for m in pool], rule, key)


def majority_correct(C):
    """Majority by correctness (section 2, used where only 0/1 is known): item solved iff sum_i C_ij > n/2
    (strict; ties for even n count as not solved). Returns accuracy."""
    C = np.asarray(C)
    return float((C.sum(0) > C.shape[0] / 2).mean())


def condorcet_homogeneous(n, p):
    """Condorcet jury theorem, independent equal p: P(Bin(n,p) > n/2) (section 3.4)."""
    return float(binom.sf(math.floor(n / 2), n, p))


def poisson_binomial_pmf(ps):
    """Distribution of the number of correct members for independent p_i (DP, as in the original analysis)."""
    dist = [1.0]
    for p in ps:
        nd = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nd[k] += v * (1 - p); nd[k + 1] += v * p
        dist = nd
    return dist


def condorcet_heterogeneous(ps):
    """Independent heterogeneous members: P(#correct > n/2) under the Poisson-binomial (section 3.4)."""
    n = len(ps)
    return sum(v for k, v in enumerate(poisson_binomial_pmf(ps)) if k > n / 2)


def median_member(ps, even="lower"):
    """rho = 1 prediction (section 3.4): majority accuracy equals the median member's accuracy.
    Odd n: sorted(desc)[(n+1)//2 - 1], the m-th best with m = (n+1)/2.
    Even n: even='lower' -> sorted(desc)[n//2], the lower of the two middle members, i.e. the
    (n/2+1)-th best. This is the rho -> 1 limit of a STRICT majority (k > n/2): at rho = 1 the items are solved in the
    same order by everyone, so more than n/2 members solve an item iff the (n/2+1)-th best does.
    even='mean' -> statistics.median, the convention of the frozen reference outputs (kept only to reproduce
    the frozen .out files in tests/test_regression.py)."""
    if even not in ("lower", "mean"):
        raise ValueError(f"median_member: even must be 'lower' or 'mean', got {even!r}")
    s = sorted(ps, reverse=True)
    n = len(s)
    if n % 2 == 1:
        return s[(n + 1) // 2 - 1]
    return median(ps) if even == "mean" else s[n // 2]


def _grid(K):
    d = [-6 + 12 * (t + 0.5) / K for t in range(K)]
    w = [_Nd.pdf(x) for x in d]
    return d, w


def copula_conditional(ps, rho, d, clamp=True):
    """p_i(D) = Phi((Phi^-1(p_i) - sqrt(rho) D) / sqrt(1 - rho)) (section 3.4). clamp: p clipped to [1e-9, 1-1e-9]
    (copula_major does, all_wrong_cop does not)."""
    out = []
    for p in ps:
        if clamp:
            p = min(max(p, 1e-9), 1 - 1e-9)
        out.append(_Nd.cdf((_Nd.inv_cdf(p) - math.sqrt(rho) * d) / math.sqrt(1 - rho)))
    return out


def copula_majority(ps, rho, K=400):
    """One-factor Gaussian copula majority (section 3.4): E_D[P(PoissonBinomial(p_i(D)) > n/2)], D ~ N(0,1), by midpoint
    rule on [-6, 6] with K nodes and weights phi(D) renormalised to sum 1 (as in the original analysis). rho >= 0.999 ->
    median_member(ps) (odd and even n). rho = 0 reduces to condorcet_heterogeneous."""
    if not (0 <= rho <= 1):
        raise ValueError(f"copula_majority: rho must be in [0, 1], got {rho}")
    if any(not (0 <= p <= 1) for p in ps):  # guard: an accuracy outside [0, 1] is an upstream bug, not something to clamp
        raise ValueError(f"copula_majority: every p must be in [0, 1], got {ps}")
    if rho >= 0.999:
        return median_member(ps)
    d, w = _grid(K)
    tot = sum(wi * condorcet_heterogeneous(copula_conditional(ps, rho, di)) for di, wi in zip(d, w))
    return tot / sum(w)


def copula_all_wrong(ps, rho, K=400):
    """Copula share of items all members fail: E_D[prod_i (1 - p_i(D))] (section 3.4 / section 4.2.1;
    no clamping of p)."""
    if not (0 <= rho < 0.999):  # guard: rho = 1 divides by zero in copula_conditional
        raise ValueError(f"copula_all_wrong: rho must be in [0, 0.999), got {rho}")
    if any(not (0 < p < 1) for p in ps):  # guard: inv_cdf(0) / inv_cdf(1) is undefined without clamping
        raise ValueError(f"copula_all_wrong: every p must be in (0, 1), got {ps}")
    d, w = _grid(K)
    tot = 0.0
    for di, wi in zip(d, w):
        pr = 1.0
        for x in copula_conditional(ps, rho, di, clamp=False):
            pr *= 1 - x
        tot += wi * pr
    return tot / sum(w)


def copula_many(P, rho, K=400, clamp=1e-9, grid="midpoint"):
    """Vectorised copula_majority and copula_all_wrong for many pools of the same size k (section 3.4): P is (pools, k)
    accuracies, rho a scalar or (pools,) latent correlations in [0, 0.999). Same model as the scalar functions; grid
    'midpoint' = their rule (midpoint on [-6, 6], K nodes, weights phi(D) renormalised), 'hermite' = K-node Gauss-Hermite
    (probabilists', weights renormalised; the public-matrix script used K = 41). p clamped to [clamp, 1 - clamp] for both outputs
    (copula_all_wrong clamps nothing, which differs only for p within 1e-9 of 0 or 1). rho = 0 gives the independent
    (Poisson-binomial) Condorcet majority and prod(1 - p). Returns (majority, all_wrong) arrays of shape (pools,)."""
    from scipy.special import ndtr, ndtri
    P = np.atleast_2d(np.asarray(P, dtype=float))
    ns, k = P.shape
    r = np.broadcast_to(np.asarray(rho, dtype=float), (ns,))
    if np.any(r < 0) or np.any(r >= 0.999):
        raise ValueError("copula_many: rho must be in [0, 0.999)")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("copula_many: every p must be in [0, 1]")
    if grid == "midpoint":
        d, w = (np.array(v) for v in _grid(K))
    elif grid == "hermite":
        d, w = np.polynomial.hermite_e.hermegauss(K)
    else:
        raise ValueError(f"copula_many: grid must be 'midpoint' or 'hermite', got {grid!r}")
    w = w / w.sum()
    th = ndtri(np.clip(P, clamp, 1 - clamp))                                            # (ns, k)
    q = ndtr((th[:, :, None] - np.sqrt(r)[:, None, None] * d[None, None, :]) / np.sqrt(1 - r)[:, None, None])  # (ns, k, K)
    dist = np.zeros((ns, k + 1, K)); dist[:, 0, :] = 1.0                               # Poisson-binomial DP per node
    for i in range(k):
        qi = q[:, i, :][:, None, :]
        new = dist * (1 - qi)
        new[:, 1:, :] += dist[:, :-1, :] * qi
        dist = new
    cnt = np.arange(k + 1)
    maj = (dist[:, cnt > k / 2, :].sum(1) * w[None, :]).sum(1)
    aw = (dist[:, 0, :] * w[None, :]).sum(1)
    return maj, aw
