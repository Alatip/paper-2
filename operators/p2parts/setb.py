"""Set B analysis — pure functions (no file I/O, no item texts). Spec: docs/SETB-SPEC.md. Used by operators/setb_report.py.

Every function is deterministic given its arguments (random draws only through explicit seeds).
"""
import collections
import hashlib
import re

import numpy as np

from p2parts import answer_id as AID

CAP = 64000          # output-token cap: an answer above it (or finish length / error) is wrong
NBOOT, NSPLIT = 10000, 20


# ---------------- answers ----------------
def norm_answer(s):
    """Normalise a LaTeX-ish final answer for explicit-equivalent matching: drop $, whitespace, \\displaystyle,
    \\left, \\right, \\!, \\, , \\; , \\( \\) and one trailing '.'; \\dfrac/\\tfrac -> \\frac; \\overline{R} -> \\overlineR;
    a leading '\\Gamma(f)=' is removed. None -> ''."""
    s = re.sub(r"\\(?:displaystyle|left|right|!|,|;)|[\s$]|\\\(|\\\)", "", s or "")
    if s.endswith("."):          # after $/space removal, so '$x = 1.$' -> 'x=1' (blind test, Fable 2026-10-26)
        s = s[:-1]
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\overline{R}", "\\overlineR")
    return re.sub(r"^\\Gamma\(f\)=", "", s)


def apply_audit(items, audit):
    """items: {id: catalogue row}; audit: {"relabel": {id: patch}, "grader_fix": {id: patch}, ...}.
    Returns (new_items, accept): new_items = copy of items where each relabel / grader_fix patch's "gold" and "gold_norm"
    keys (only those) overwrite the row; accept = {id: set(norm_answer(x) for x in patch["accept"])} for grader_fix patches
    that have an "accept" list. The input dict is not modified."""
    new = dict(items)
    for i, p in {**audit.get("relabel", {}), **audit.get("grader_fix", {})}.items():
        if i in new:
            new[i] = {**new[i], **{k: v for k, v in p.items() if k in ("gold", "gold_norm")}}
    accept = {i: {norm_answer(x) for x in p["accept"]} for i, p in audit.get("grader_fix", {}).items() if p.get("accept")}
    return new, accept


def answer_hash(s):
    """sha256 hex of norm_answer(s) (UTF-8) — how audited accepted answers are published without revealing them."""
    return hashlib.sha256(norm_answer(s).encode("utf-8")).hexdigest()


def over_cap(row, cap=CAP):
    """True if the answer reached the output cap: tok_out >= cap (rule of 2026-10-29; before it only tok_out > cap counted,
    so claude CLI answers that stopped at exactly the cap after an extra turn were graded)."""
    return (row.get("tok_out") or 0) >= cap


def is_correct(row, grade_fn, accept=None, parse_fn=None, cap=CAP, hashed=False):
    """1 if the answer row is correct, else 0. row None, finish in {length, error} or tok_out >= cap -> 0.
    If accept is given (a set of normalised answers, or of answer_hash values when hashed=True): 1 iff
    norm_answer(parse_fn(text)) (resp. answer_hash(parse_fn(text))) is in it; grade_fn is not called.
    Else: int(bool(grade_fn(row))) (grade_fn returns truthy for correct)."""
    if not row or row.get("finish") in ("length", "error") or over_cap(row, cap):
        return 0
    if accept is not None:
        a = parse_fn(row.get("text") or "")
        return int((answer_hash(a) if hashed else norm_answer(a)) in accept)
    return int(bool(grade_fn(row)))


def plurality(answers, oks):
    """Paper-1 plurality over one vote. answers: list of answer strings (None/'' = no answer); oks: 0/1 per answer.
    Returns (correct 0/1, winning group key or None, keys per answer). Identity = answer_id (RULE p1); ties 'first'
    (the tied group whose first member comes first wins); a group is correct if any member is correct; no answers -> 0."""
    top, _, okv, k = AID.vote([a or None for a in answers], list(oks))
    return int(bool(okv)), top, k


def is_unanimous_by_identity(keys):
    """True iff exactly one non-None group key occurs (all answering members in one paper-1 group)."""
    return len({x for x in keys if x is not None}) == 1


def new_correct_answer(op_answer, op_ok, input_answers):
    """1 iff the operator's answer is correct and its paper-1 group contains none of the input answers."""
    if not op_ok or not op_answer:
        return 0
    k = AID.keys([a or None for a in input_answers] + [op_answer])
    return int(k[-1] is not None and k[-1] not in set(k[:-1]))


def plurality_codes(codes, oks):
    """Plurality over opaque answer codes (public data: equal code = same paper-1 group; '' or None = no answer).
    Winner = most members; tie -> the group whose first member comes first. Returns (correct 0/1, winning code or None):
    correct = 1 iff any member of the winning group has ok 1; no answers -> (0, None)."""
    counts, first = collections.Counter(), {}
    for j, c in enumerate(codes):
        if c:
            counts[c] += 1
            first.setdefault(c, j)
    if not counts:
        return 0, None
    top = min(counts, key=lambda c: (-counts[c], first[c]))
    return int(any(o for c, o in zip(codes, oks) if c == top)), top


def new_correct_code(op_code, op_ok, input_codes):
    """1 iff op_ok is truthy, op_code is non-empty and differs from every input code; else 0."""
    return int(bool(op_ok) and bool(op_code) and op_code not in set(input_codes))


# ---------------- cost ----------------
def row_cost(row, model, list_prices, cap=CAP):
    """$ of one call. None row -> (0.0, False). OpenRouter (model not starting with 'claude-'): (row['cost'] or 0, False).
    Claude: (cost_notional, False) when present; else an estimate (pi_in * tok_in + pi_out * cap, True) with
    (pi_in, pi_out) = list_prices[model] ($ per token)."""
    if not row:
        return 0.0, False
    if model.startswith("claude-"):
        if row.get("cost_notional") not in (None, ""):
            return float(row["cost_notional"]), False
        a, b = list_prices[model]
        return a * float(row.get("tok_in") or 0) + b * cap, True
    return float(row.get("cost") or 0), False


# ---------------- statistics ----------------
def operator_seed(name):
    """Bootstrap seed of an operator: first 8 hex digits of sha256(name) as int."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def bootstrap_ci(d, rng, n_boot=NBOOT):
    """Paired item bootstrap of the mean of d: [mean, 2.5 %, 97.5 %] over n_boot resamples (indices drawn by
    rng.integers(0, n, size=(n_boot, n))). Consumes rng."""
    d = np.asarray(d, float); n = len(d)
    b = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def splits(strata, n_split=NSPLIT, seed=0):
    """n_split boolean masks (True = half 0) over len(strata) items, stratified: within each stratum (sorted order of
    distinct labels) a shuffled index list has its first floor(size/2) entries set True. rng = default_rng(seed)."""
    g = np.random.default_rng(seed); lab = np.asarray(strata); out = []
    for _ in range(n_split):
        m = np.zeros(len(lab), bool)
        for L in sorted(set(lab)):
            idx = np.flatnonzero(lab == L); g.shuffle(idx); m[idx[: len(idx) // 2]] = True
        out.append(m)
    return out


def heldout_best(P, S):
    """P: items x participants 0/1. For each mask and each direction (choose on one half, score on the other), the
    participant with the highest mean on the choice half (first on ties) is scored on the other half. Returns the
    per-item value averaged over masks (each item is scored once per mask)."""
    P = np.asarray(P, float); v = np.zeros(P.shape[0])
    for m in S:
        for a, b in ((m, ~m), (~m, m)):
            v[b] += P[b, int(np.argmax(P[a].mean(0)))]
    return v / len(S)


def heldout_best_under_cost(acc, cost, CA, S, names):
    """acc, cost: {name: per-item array}; CA: per-item operator cost. On each choice half: candidates = names whose mean
    cost <= mean CA on that half; pick max (mean acc, -mean cost, name); score acc on the other half. If some half has no
    candidate -> (None, picks so far). Returns (per-item value averaged over masks, Counter of picks)."""
    CA = np.asarray(CA, float); v = np.zeros(len(CA)); picks = collections.Counter()
    for m in S:
        for a, b in ((m, ~m), (~m, m)):
            cand = [(acc[k][a].mean(), -cost[k][a].mean(), k) for k in names if cost[k][a].mean() <= CA[a].mean()]
            if not cand:
                return None, picks
            k = max(cand)[2]; v[b] += acc[k][b]; picks[k] += 1
    return v / len(S), picks


def resample_headroom(P, acc):
    """Resampling operators. P: items x draws 0/1; acc: operator accuracy (mean). Returns {mean_draw, pass_at_k,
    headroom = pass_at_k - mean_draw, share = (acc - mean_draw) / headroom or None if headroom <= 0}."""
    P = np.asarray(P, float); md, pk = float(P.mean()), float(P.max(1).mean())
    return {"mean_draw": md, "pass_at_k": pk, "headroom": pk - md, "share": (float((acc - md) / (pk - md)) if pk > md else None)}


def operator_stats(name, a, a0, P, CA, C0, acc, cost, strata, names, mcnemar, resample=False):
    """All reported numbers of one operator on its items (arrays aligned by item).
    a, a0: 0/1 operator / starting point; P: items x participants 0/1; CA, C0: $ per item of operator / starting point;
    acc, cost: {name: per-item array} single models on the same items; strata: label per item; mcnemar(W, L) -> p.
    rng = default_rng(operator_seed(name)), used in this order: delta CI, then gain-vs-best(C) CI.
    Keys: N, acc, acc0, delta [m, lo, hi], W (a0=0,a=1), L (a0=1,a=0), mcnemar_p, C_usd, C0_usd,
    usd_per_extra_item ((mean CA - mean C0)/delta if delta > 0 else None); if resample: resample = resample_headroom;
    if P has > 1 column: Cplus (all participants 0 and a=1), oracle (mean of row max), best_heldout, best_insample
    (max column mean), prize = oracle - best_heldout, s_heldout = (acc - best_heldout)/prize or None;
    bestC_heldout = None or {acc, picks, gain = CI of a - v}.
    Returns (out, rng): rng is returned so extra CIs of the same operator continue the same stream."""
    rng = np.random.default_rng(operator_seed(name))
    a, a0, CA, C0 = (np.asarray(x, float) for x in (a, a0, CA, C0)); P = np.asarray(P, float)
    S = splits(strata)
    d = bootstrap_ci(a - a0, rng)
    W = int(((a0 == 0) & (a == 1)).sum()); L = int(((a0 == 1) & (a == 0)).sum())
    out = {"N": len(a), "acc": float(a.mean()), "acc0": float(a0.mean()), "delta": d, "W": W, "L": L,
           "mcnemar_p": float(mcnemar(W, L)), "C_usd": float(CA.mean()), "C0_usd": float(C0.mean()),
           "usd_per_extra_item": float((CA.mean() - C0.mean()) / d[0]) if d[0] > 0 else None}
    if resample:
        out["resample"] = resample_headroom(P, float(a.mean()))
    if P.shape[1] > 1:
        allw = P.max(1) == 0
        oracle = float(P.max(1).mean()); bho = float(heldout_best(P, S).mean()); bin_ = float(P.mean(0).max())
        out.update(Cplus=int((allw & (a == 1)).sum()), oracle=oracle, best_heldout=bho, best_insample=bin_,
                   prize=oracle - bho, s_heldout=(float((a.mean() - bho) / (oracle - bho)) if oracle > bho else None))
    v, picks = heldout_best_under_cost(acc, cost, CA, S, names)
    out["bestC_heldout"] = None if v is None else {"acc": float(v.mean()), "picks": dict(picks), "gain": bootstrap_ci(a - v, rng)}
    return out, rng
