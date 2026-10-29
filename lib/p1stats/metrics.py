"""Accuracy, oracle, best member, prize, collected share; in-sample and held-out (split-half) variants.

Paper 1, section 2 (definitions: oracle / coverage, best member, prize = oracle - best, share s) and section 3.3 (router
ceiling), section 4 (split-half protocol). Inputs are a correctness matrix C (models x items, 0/1) unless stated.

Tie rule (in-sample and on a training half): the best member is the FIRST model in pool order among those with maximal
accuracy (Python max(pool, key=...))."""
import random

import numpy as np


def accuracy(c):
    """acc = (1/N) sum_j c_j for a 0/1 vector (section 2)."""
    c = np.asarray(c, dtype=float)
    return float(c.mean())


def oracle(C):
    """Oracle (coverage, pass@n over members): share of items solved by at least one member,
    (1/N) sum_j max_i C_ij (section 2)."""
    C = np.asarray(C, dtype=float)
    return float(C.max(0).mean())


def best_index(C):
    """Index of the in-sample best member: argmax_i acc_i, ties -> first in row (pool) order (section 2)."""
    acc = np.asarray(C, dtype=float).mean(1)
    return int(np.flatnonzero(acc == acc.max())[0])


def best(C):
    """In-sample best-member accuracy max_i acc_i (section 2)."""
    return float(np.asarray(C, dtype=float).mean(1).max())


def prize(C):
    """Prize = oracle - best (section 2): what a perfect per-item selector could add over the best member."""
    return oracle(C) - best(C)


def collected_share(acc_op, best_acc, oracle_acc):
    """Collected share s = (acc - best)/(oracle - best) (section 2); nan when oracle == best."""
    d = oracle_acc - best_acc
    return float("nan") if d <= 0 else (acc_op - best_acc) / d


def all_wrong(C):
    """Number of items no member solves (section 4.2, 'all-wrong')."""
    return int((np.asarray(C).sum(0) == 0).sum())


def split_half(ids, rng):
    """One random half split as in the frozen reference outputs: copy ids, rng.shuffle (random.Random), A = first N//2, B = rest."""
    sh = list(ids)
    rng.shuffle(sh)
    h = len(sh) // 2
    return sh[:h], sh[h:]


def pct_2000(sorted_vals):
    """Percentile convention for 2,000 resamples: (v[50], v[1949]) of the sorted list,
    generalised to (v[int(0.025 B)], v[int(0.975 B) - 1])."""
    B = len(sorted_vals)
    return sorted_vals[int(0.025 * B)], sorted_vals[int(0.975 * B) - 1]


def heldout_best_minus(score_best, score_op, ids, pool, n_splits=2000, seed=0, rng=None, keep=None, dropped=None,
                       op_uses_member=False, reshuffle="copy", interval="pct2000"):
    """Split-half held-out comparison (section 4.2, 'split best-maj'): best member chosen on half A by
    sum of score_best(i, m) (ties: first in pool order), then mean over B of score_best(i, bm) - score_op(i).
    keep(i, bm) -> bool restricts B after the choice ('answered-only'): items where bm answered);
    the number of dropped B items per split is appended to the list `dropped` if given.
    op_uses_member=True: score_op is called as score_op(i, bm) (panel - best: off the judged items the operator
    falls back to the chosen member's own correctness).
    reshuffle: 'copy' = every split shuffles a fresh copy of ids (split_half; corpus scripts); 'cumulative' = one list
    shuffled again in place at every split (the SWE-bench window script); both draw A = first N//2.
    interval: 'pct2000' = pct_2000 of the sorted differences; 'numpy' = np.percentile 2.5 / 97.5 (linear interpolation).
    Returns (mean, lo, hi) over n_splits splits; rng = random.Random(seed) unless given."""
    if reshuffle not in ("copy", "cumulative") or interval not in ("pct2000", "numpy"):
        raise ValueError(f"heldout_best_minus: bad reshuffle={reshuffle!r} or interval={interval!r}")
    rng = rng or random.Random(seed)
    ds = []
    cum = list(ids)
    for _ in range(n_splits):
        if reshuffle == "copy":
            A, B = split_half(ids, rng)
        else:
            rng.shuffle(cum); h = len(cum) // 2; A, B = cum[:h], cum[h:]
        bm = max(pool, key=lambda m: sum(score_best(i, m) for i in A))
        if keep is not None:
            Bk = [i for i in B if keep(i, bm)]
            if dropped is not None:
                dropped.append(len(B) - len(Bk))
            B = Bk
        if not B:  # guard: keep() removed every item of half B
            raise ValueError("heldout_best_minus: half B is empty after keep()")
        so = (lambda i: score_op(i, bm)) if op_uses_member else score_op
        ds.append((sum(score_best(i, bm) for i in B) - sum(so(i) for i in B)) / len(B))
    ds.sort()
    lo, hi = pct_2000(ds) if interval == "pct2000" else (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
    return sum(ds) / len(ds), lo, hi


def routing_in_sample(X, pool, layers):
    """Per-set router ceiling in sample (section 3.3; routing.py run()).
    X: {layer: {item: {model: 0/1}}}. Returns best model over all items, its accuracy, route {layer: winner},
    routed accuracy, oracle. Ties: first in pool order."""
    allids = [(L, i) for L in layers for i in X[L]]
    N = len(allids)
    tot = {m: sum(X[L][i][m] for L, i in allids) for m in pool}
    b = max(pool, key=lambda m: tot[m])
    route, rc = {}, 0
    for L in layers:
        w = max(pool, key=lambda m: sum(X[L][i][m] for i in X[L]))
        route[L] = w
        rc += sum(X[L][i][w] for i in X[L])
    orc = sum(max(X[L][i][m] for m in pool) for L, i in allids) / N
    return b, tot[b] / N, route, rc / N, orc


def routing_split_half(X, pool, layers, n_splits=2000, seed=0):
    """Held-out router (section 3.3): ONE random.Random(seed) stream; for every split each layer's ids
    are shuffled in layer order and halved (h = n//2); best overall and per-layer winner chosen on A, scored on B.
    Returns dict mean best/route/oracle, mean route-best and (lo, hi) by the 2,000-convention, share (mr-mb)/(mo-mb)."""
    rng = random.Random(seed)
    ds, bs, rs, os_ = [], [], [], []
    for _ in range(n_splits):
        A, B = {}, {}
        for L in layers:
            ids = list(X[L]); rng.shuffle(ids); h = len(ids) // 2; A[L] = ids[:h]; B[L] = ids[h:]
        tot = {m: sum(X[L][i][m] for L in layers for i in A[L]) for m in pool}
        bm = max(pool, key=lambda m: tot[m])
        win = {L: max(pool, key=lambda m: sum(X[L][i][m] for i in A[L])) for L in layers}
        nb = sum(len(B[L]) for L in layers)
        bb = sum(X[L][i][bm] for L in layers for i in B[L]) / nb
        rr = sum(X[L][i][win[L]] for L in layers for i in B[L]) / nb
        oo = sum(max(X[L][i][m] for m in pool) for L in layers for i in B[L]) / nb
        bs.append(bb); rs.append(rr); os_.append(oo); ds.append(rr - bb)
    ds.sort()
    mb, mr, mo = [sum(v) / len(v) for v in (bs, rs, os_)]
    lo, hi = pct_2000(ds)
    return dict(best=mb, route=mr, oracle=mo, diff=sum(ds) / len(ds), lo=lo, hi=hi, share=collected_share(mr, mb, mo))  # nan if the pool has no prize at all (oracle == best on every half B)


def median_vs_heldout_best(B, subsets, seeds=20):
    """rho = 1 prediction check (section 3.4): for each seed s, items permuted by
    np.random.default_rng(s), train = first N//2; per subset order members by TRAIN accuracy (np.argsort(-acc, kind="stable"), stable
    ties by index), best = order[0], median = order[(k-1)//2]; returns mean over seeds x subsets of test(median) - test(best).
    B: items x models 0/1."""
    B = np.asarray(B, dtype=float)
    gains = []
    for seed in range(seeds):
        perm = np.random.default_rng(seed).permutation(B.shape[0]); h = B.shape[0] // 2
        atr, ate = B[perm[:h]].mean(0), B[perm[h:]].mean(0)
        for sub in subsets:
            sub = np.asarray(sub)
            order = sub[np.argsort(-atr[sub], kind="stable")]  # stable: ties by index, as promised above (default quicksort is not stable for k >= 17)
            k = len(sub)
            gains.append(ate[order[(k - 1) // 2]] - ate[order[0]])
    return float(np.mean(gains))


def router_stochastic(P):
    """Section 3.3 router with random answers (R*). P: models x items, per-item success
    probabilities. Returns (best = max_i mean_q p_i, R* = mean_q max_i p_i, expected one-run oracle = mean_q [1 - prod_i(1-p_i)])."""
    P = np.asarray(P, dtype=float)
    return float(P.mean(1).max()), float(P.max(0).mean()), float((1 - np.prod(1 - P, axis=0)).mean())
