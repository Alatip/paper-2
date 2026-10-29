"""median_vs_heldout_best must order tied members by index for any subset size (np.argsort default is not stable
for >= 17 elements). Review 2026-10-17."""
import numpy as np

from p1stats import metrics as M


def _reference(B, subsets, seeds):
    """Independent reference: order by (-train accuracy, index) with Python's sorted()."""
    gains = []
    for seed in range(seeds):
        perm = np.random.default_rng(seed).permutation(B.shape[0]); h = B.shape[0] // 2
        atr, ate = B[perm[:h]].mean(0), B[perm[h:]].mean(0)
        for sub in subsets:
            order = sorted(sub, key=lambda m: (-atr[m], m))
            gains.append(ate[order[(len(sub) - 1) // 2]] - ate[order[0]])
    return float(np.mean(gains))


def test_median_ties_by_index_large_k():
    # 24 members built from 3 column types -> many exact ties in train accuracy; subsets of size 20 (>= 17)
    rng = np.random.default_rng(7)
    cols = rng.integers(0, 2, size=(60, 3)).astype(float)
    B = cols[:, rng.integers(0, 3, size=24)]
    B[:, 5] = rng.integers(0, 2, size=60)            # one member different on test, so the choice matters
    subsets = [list(range(20)), list(range(4, 24))]
    assert abs(M.median_vs_heldout_best(B, subsets, seeds=10) - _reference(B, subsets, 10)) < 1e-12
