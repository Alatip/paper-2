"""Operator "budget": operators at equal budget in tokens and dollars, and the break-even algebra (paper v8 §10.1–10.7, tables 10.1, 10.2;
cost-matched Q1–Q5).

Port of a private script (not published) (Q1–Q5) + the numbers render.py/§10 read off results.json. Reads only
datasets/budget-* through p2parts/budget.py. $0, offline, no model calls.

Method (as run.py):
  Q1  one DeepSeek V3.2 answer (T = 0) cut at B ∈ {256..8192} tokens: lower bracket (cut = wrong) and upper bracket (last parsable answer in
      the cut text), vs the budget-forcing protocol (STATS G) at the same B.
  Q2  16 T = 1 draws per item; n draws each cut at B/n, plurality (ties wrong) vs one draw cut at B (mean over 16 draws), paired gain.
  Q3  operators at their own cost (k draws plurality, six-copies debate, six-families debate; last round, finish=length dropped) vs the
      natural T = 0 answer.
  Q4  plurality of k Haiku draws vs one Opus / Sonnet 5.5 answer, nominal API cost per item.
  Q5  one-factor Gaussian copula: ρ fitted so that model φ == mean observed pairwise φ of the 16 draws; P_maj(p_c, ρ, n) − p_c.
  Intervals: 95% item bootstrap, 2000 draws, ONE numpy Generator(seed 0) consumed in run.py's exact call order (Q1 → Q4), so CIs
  reproduce the stored results.json bit-for-bit only if datasets and call order are unchanged.

Implementation decisions:
  - The cut at B needs the trace text (tiktoken cl100k proportional cut); texts are not stored. We read run.py's own text-free export
    (cut-correct.jsonl: per trace and per B the lower/upper correctness, cut flag and an anonymised vote label). Natural (uncut) vote
    labels for Q3/Q4 come from build_budget.py, which grades with run.py's grader. Labels are injective per item, so pluralities equal
    run.py's.
  - Answer identity (decision 2026-10-18, switched 2026-10-18): Q3 (k draws, both debates) and Q4 (k Haiku draws) vote with paper 1's
    rule (p2parts/answer_id.py) over the stored answer keys (akey: letters / opaque ~k codes for MMLU-Pro and GPQA, the parsed answer
    elsewhere; rebuilt by a maintainer build script (not published)), a group correct if any member is graded correct.
    P2_ANSWER_RULE=v8 = run.py's labels (GOLD / W<k>, gnorm identity). Q2 still votes on run.py's labels of the CUT traces
    (cut-correct.jsonl): the cut needs the trace texts, which are not in datasets/. The tie rule (a tie = wrong) is v8's in both modes.
  - Q3 natural T=0 answer (P2_MATRIX=corpus, default): correctness and output tokens from paper 1's uncapped matrix for every item
    present there (label audits as in paper 1); P2_MATRIX=tn5 = the stored capped row. Q1 (cut texts) is unchanged.
  - "s10.*" keys: the quantities §10 prints (ranges, ratios, best-n) computed from the Q1–Q5 tables; their definitions are in the
    function docstrings (they were read off RESULTS.md by hand in v8).
  - B/natural (table 10.1) divides by Q2's mean tokens per T = 1 draw (2048/227.8 = 9.0 etc.), not Q1's T = 0 mean; the §10.4
    families-debate token range (21–32×) is over MMLU-Pro, GPQA, GSM-Hard (olympiad 12.4× is outside it).
  - Opus thinking tokens (§10.5 "54–128"): reported both over the Q4 items (s10.10_5.opus_think_*) and over all Opus rows of the layer
    (opus_think_all_rows_*); neither reproduces 54–128 exactly (see notes).
  - "untraced.*": n_eff(9) (Kish), 1/φ̄ and the n → ∞ copula limit of §10.6 — new functions, see untraced().
  - Dollar budget (decided 2026-10-19; P2_BUDGET_UNIT=usd, default): key "dollar" = compute_dollar(): table 10.1 and §10.4
    redone with C(A) in dollars (datasets/model-prices) against best(C(A)) = the most accurate natural single answer of the 7-model
    pool costing <= C(A). P2_BUDGET_UNIT=tokens = v8 only (equal-token Q1–Q5; pinned for the regressions). Details in compute_dollar().
"""
import collections
import math
import os

import numpy as np
from scipy.stats import binom, norm as N

from p2parts import budget as IO
from p2parts import answer_id as AID

# Q3 natural T=0 answer: "corpus" (default, 2026-10-18) = paper 1's uncapped matrix where the item is in it; "tn5" = the stored
# capped DeepSeek T=0 row (v8). Env P2_MATRIX, read at import (or set MATRIX_SOURCE).
MATRIX_SOURCE = os.environ.get("P2_MATRIX", "corpus")
assert MATRIX_SOURCE in ("corpus", "tn5"), MATRIX_SOURCE

LAYERS = IO.LAYERS
BUD = [256, 512, 1024, 2048, 4096, 8192]
Q2B = [1024, 2048, 4096, 8192, 16384, 32768]
NS = [1, 3, 5, 7, 9, 15]


class Boot:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, vals, n=2000):
        v = np.asarray(vals, float); idx = self.rng.integers(0, len(v), (n, len(v)))
        m = v[idx].mean(1); return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def plurality(answers):
    a = [x for x in answers if x is not None]
    if not a: return None
    c = collections.Counter(a).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]: return "TIE"
    return c[0][0]


def right(a): return a == "GOLD"


def vote_right(rows, field="vote"):
    """Is the plurality of these rows correct? Tie = no answer = wrong (v8 rule, kept). AID.RULE == "v8": run.py's labels
    (GOLD / W<k>). "p1" (default): paper 1's answer identity over the rows' akey (answer_id.keys: 12-digit canon, decimal rounding,
    chained groups); a group is correct if any member is graded correct (row 'ok')."""
    if AID.RULE == "v8":
        return right(plurality([r[field] for r in rows]))
    rows = [r for r in rows if r.get("akey")]
    if not rows: return False
    k = AID.keys([r["akey"] for r in rows])
    c = collections.Counter(k).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]: return False
    top = c[0][0]
    return any(r["ok"] for x, r in zip(k, rows) if x == top)


def compute_q(boot=None):
    boot = boot or Boot(0)
    cut = IO.cut_rows(); nat = IO.natural(); forcing = IO.forcing(); deb = IO.debates(); cl = IO.claude()
    res = {}
    # Q1
    bf = {}
    for r in forcing: bf.setdefault((r["layer"], r["budget"]), []).append(r["correct"])
    q1 = {}
    for L in LAYERS:
        rows = [r for r in cut if r["layer"] == L and r["stage"] == "single"]
        q1[L] = {"n": len(rows), "natural_acc": boot([r["correct"] for r in rows]), "mean_tok": float(np.mean([r["out_tok"] for r in rows])),
                 "B": {}}
        for B in BUD:
            c = [r["cut"][str(B)] for r in rows]
            q1[L]["B"][str(B)] = {"lower": boot([x["lower"] for x in c]), "upper": boot([x["upper"] for x in c]),
                                  "cut_share": float(np.mean([x["is_cut"] for x in c])),
                                  "budget_forcing": boot(bf[(L, B)]) if (L, B) in bf else None}
    res["q1"] = q1
    # Q2
    q2, samples = {}, {}
    for L in LAYERS:
        by = collections.defaultdict(dict)
        for r in cut:
            if r["layer"] == L and r["stage"] == "samples": by[r["id"]][r["sample"]] = r
        ids = sorted(i for i, d in by.items() if len(d) >= 16)
        samples[L] = (ids, by)
        q2[L] = {"n_items": len(ids), "mean_tok_per_draw": float(np.mean([by[i][s]["out_tok"] for i in ids for s in range(16)])), "B": {}}
        for B in Q2B:
            sing_u = [np.mean([by[i][s]["cut"][str(B)]["upper"] for s in range(16)]) for i in ids]
            sing_l = [np.mean([by[i][s]["cut"][str(B)]["lower"] for s in range(16)]) for i in ids]
            row = {"single_upper": boot(sing_u), "single_lower": boot(sing_l), "vote": {}}
            for n in NS[1:]:
                vu, vl = [], []
                for i in ids:
                    bs = [by[i][s]["cut"][str(B // n)] for s in range(n)]
                    vu.append(right(plurality([b["vote"] for b in bs])))
                    vl.append(right(plurality([b["vote"] for b in bs if not b["is_cut"]])))
                gain = np.array(vu, float) - np.array(sing_u, float)
                row["vote"][str(n)] = {"upper": boot(vu), "lower": boot(vl), "gain_upper": boot(gain)}
            q2[L]["B"][str(B)] = row
    res["q2"] = q2
    # Q3
    q3 = {}
    for L in LAYERS:
        ids, _ = samples[L]
        single = {r["id"]: r for r in nat if r["layer"] == L and r["stage"] == "single"}
        cm = {}
        if MATRIX_SOURCE == "corpus":   # natural T=0 answer from paper 1's uncapped matrix (correctness and output tokens)
            cm = IO.corpus_deepseek(L)
            for i, r in list(single.items()):
                if i in cm: single[i] = dict(r, correct=bool(cm[i]["correct"]), out_tok=cm[i]["out_tok"])
        nby = collections.defaultdict(dict)
        for r in nat:
            if r["layer"] == L and r["stage"] == "samples" and r["out_tok"]: nby[r["id"]][r["sample"]] = r
        ids3 = [i for i in ids if i in single and single[i]["out_tok"]]
        s1 = np.array([single[i]["correct"] for i in ids3], float)
        out = {"n_items": len(ids3), "single_from_corpus": sum(1 for i in ids3 if i in cm), "single": {"acc": boot(s1), "tok": float(np.mean([single[i]["out_tok"] for i in ids3]))}, "draws": {}}
        for k in range(1, 17):
            acc = [vote_right([nby[i][s] for s in range(k)]) for i in ids3]
            out["draws"][str(k)] = {"acc": boot(acc), "gain": boot(np.array(acc, float) - s1),
                                    "tok": float(np.mean([sum(nby[i][s]["out_tok"] for s in range(k)) for i in ids3]))}
        for name, kind in (("copies_debate", "copies"), ("families_debate", "families")):
            src = collections.defaultdict(list)
            for r in deb:
                if r["layer"] == L and r["kind"] == kind and r["id"] in ids3: src[r["id"]].append(r)
            ok = [i for i in ids3 if src[i]]
            if not ok: continue

            def final(rs):
                last = max(r["round"] for r in rs)
                return vote_right([r for r in rs if r["round"] == last and r["finish"] != "length"])
            acc = [final(src[i]) for i in ok]
            out[name] = {"n": len(ok), "acc": boot(acc), "tok": float(np.mean([sum((r["out_tok"] or 0) for r in src[i]) for i in ok])),
                         "gain": boot(np.array(acc, float) - np.array([single[i]["correct"] for i in ok], float))}
        q3[L] = out
    res["q3"] = q3
    # Q4
    q4 = {}
    for L in LAYERS:
        hb = collections.defaultdict(dict)
        for r in cl:
            if r["model"] == "claude_haiku" and r["layer"] == L and r["stage"] == "samples": hb[r["id"]][r["sample"]] = r
        comp = {m: {} for m in ("claude_sonnet55", "claude_opus", "claude_haiku")}
        for r in cl:
            if r["layer"] == L and r["stage"] == "single": comp[r["model"]][r["id"]] = r
        kmax = min(len(d) for d in hb.values()) if hb else 0
        ids = sorted(i for i in hb if all(i in comp[m] for m in comp) and len(hb[i]) >= kmax)
        out = {"n_items": len(ids), "k_available": kmax, "comparators": {}, "haiku": {}}
        for m in comp:
            out["comparators"][m] = {"acc": boot([comp[m][i]["correct"] for i in ids]), "cost_per_item": float(np.mean([comp[m][i]["cost"] or 0 for i in ids])),
                                     "think_tok_mean": float(np.mean([comp[m][i]["think_tok"] or 0 for i in ids]))}
        for k in [k for k in (1, 3, 5, 9, 15) if k <= kmax]:
            acc = [vote_right([hb[i][s] for s in sorted(hb[i])[:k]]) for i in ids]
            out["haiku"][str(k)] = {"acc": boot(acc), "cost_per_item": float(np.mean([sum(hb[i][s]["cost"] or 0 for s in sorted(hb[i])[:k]) for i in ids]))}
        q4[L] = out
    res["q4"] = q4
    # Q5
    q5 = {}
    for L in LAYERS:
        ids, by = samples[L]
        M = np.array([[by[i][s]["correct"] for s in range(16)] for i in ids], float)
        p = float(M.mean()); ph = phi_pairs(M); rho = fit_rho(p, ph)
        q5[L] = {"p_draw": p, "phi": ph, "rho_tetra": rho,
                 "gain_pts": {str(n): {k: (pmaj(pc, rho, n) - pc) * 100 for k, pc in (("0.6", .6), ("0.75", .75), ("0.9", .9), (repr(p), p))}
                              for n in (3, 5, 9)}}
    res["q5"] = q5
    res["_aux"] = {"olympiad_single_finish": collections.Counter(str(r["finish"]) for r in nat if r["layer"] == "olympiad" and r["stage"] == "single"),
                   "opus_think_all_rows": {L: float(np.mean([r["think_tok"] or 0 for r in cl if r["model"] == "claude_opus" and r["layer"] == L
                                                             and r["stage"] == "single"])) for L in LAYERS},
                   "olympiad_single_n": sum(1 for r in nat if r["layer"] == "olympiad" and r["stage"] == "single")}
    return res


# ---------- copula (run.py Q5) ----------
def pmaj(p, rho, n, G=4000):
    """P(majority of n correct) when each copy is correct w.p. p and errors follow a one-factor probit with latent correlation rho."""
    tau = N.ppf(p); z = np.linspace(-6, 6, G); w = N.pdf(z) * (z[1] - z[0])
    q = N.cdf((tau + math.sqrt(rho) * z) / math.sqrt(1 - rho)); k = n // 2 + 1
    return float((w * binom.sf(k - 1, n, q)).sum())


def phi_pairs(M):
    ph = []
    for a in range(M.shape[1]):
        for b in range(a + 1, M.shape[1]):
            x, y = M[:, a], M[:, b]
            if x.std() and y.std(): ph.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(ph))


def fit_rho(p, ph, G=4000):
    def model_phi(rho):
        tau = N.ppf(p); z = np.linspace(-6, 6, G); w = N.pdf(z) * (z[1] - z[0]); q = N.cdf((tau + math.sqrt(rho) * z) / math.sqrt(1 - rho))
        return (float((w * q * q).sum()) - p * p) / (p * (1 - p))
    lo, hi = 0.0, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        if model_phi(mid) < ph: lo = mid
        else: hi = mid
    return (lo + hi) / 2


# ---------- untraced §10.6 ----------
def kish_neff(n, phi):
    """Kish effective number of independent votes among n equicorrelated votes: n / (1 + (n − 1) φ)."""
    return n / (1 + (n - 1) * phi)


def copula_limit(pc, rho):
    """n → ∞ limit of P_maj(p_c, ρ, n) − p_c in the one-factor copula: Φ(Φ⁻¹(p_c)/√ρ) − p_c (the §10.6 formula)."""
    return float(N.cdf(N.ppf(pc) / math.sqrt(rho)) - pc)


def untraced(q5):
    phis = {L: q5[L]["phi"] for L in LAYERS}; rhos = {L: q5[L]["rho_tetra"] for L in LAYERS}
    grid = np.linspace(0.5005, 0.9995, 1999)
    lim = {L: {"at_0.6": copula_limit(.6, rhos[L]), "at_0.75": copula_limit(.75, rhos[L]), "at_0.9": copula_limit(.9, rhos[L]),
               "at_p_draw": copula_limit(q5[L]["p_draw"], rhos[L])} for L in LAYERS}
    for L in LAYERS:
        v = [copula_limit(pc, rhos[L]) for pc in grid]; j = int(np.argmax(v))
        lim[L]["sup_over_pc"] = float(v[j]); lim[L]["argmax_pc"] = float(grid[j])
    return {"neff9": {L: kish_neff(9, phis[L]) for L in LAYERS},
            "neff9_min": min(kish_neff(9, x) for x in phis.values()), "neff9_max": max(kish_neff(9, x) for x in phis.values()),
            "inv_phi": {L: 1 / phis[L] for L in LAYERS},
            "inv_phi_min": min(1 / x for x in phis.values()), "inv_phi_max": max(1 / x for x in phis.values()),
            "limit_pts": {L: {k: 100 * v for k, v in d.items() if k != "argmax_pc"} | {"argmax_pc": d["argmax_pc"]} for L, d in lim.items()},
            "limit_pts_max_grid3": 100 * max(lim[L][k] for L in LAYERS for k in ("at_0.6", "at_0.75", "at_0.9")),
            "limit_pts_max_sup": 100 * max(lim[L]["sup_over_pc"] for L in LAYERS)}


# ---------- numbers §10 prints ----------
def section10(r):
    q1, q2, q3, q4, q5 = r["q1"], r["q2"], r["q3"], r["q4"], r["q5"]
    s = {}
    # 10.2: budget forcing minus native cut (upper == lower bracket here) at B
    d = {L: {B: q1[L]["B"][B]["budget_forcing"][0] - q1[L]["B"][B]["upper"][0] for B in ("1024", "2048", "4096")} for L in LAYERS}
    mg = [d[L][B] for L in ("mmlu-pro", "gpqa-diamond") for B in d[L]]
    s["10_2"] = {"n": {L: q1[L]["n"] for L in LAYERS}, "bf_minus_cut": d,
                 "mmlu_gpqa_gap_min": min(mg), "mmlu_gpqa_gap_max": max(mg),
                 "olymp_gap_min": min(abs(x) for x in d["olympiad"].values()), "olymp_gap_max": max(abs(x) for x in d["olympiad"].values()),
                 "natural_acc": {L: q1[L]["natural_acc"][0] for L in LAYERS},
                 "mmlu_gpqa_gap_min_pts": 100 * min(mg), "mmlu_gpqa_gap_max_pts": 100 * max(mg),
                 "olymp_gap_min_pts": 100 * min(abs(x) for x in d["olympiad"].values()),
                 "olymp_gap_max_pts": 100 * max(abs(x) for x in d["olympiad"].values())}
    # 10.3 / table 10.1: in-sample best n per (layer, B) = n with the largest mean paired gain (upper bracket)
    best, neg, allg = {}, [], []
    for L in LAYERS:
        best[L] = {}
        for B, row in q2[L]["B"].items():
            n = max(row["vote"], key=lambda k: row["vote"][k]["gain_upper"][0]); g = row["vote"][n]["gain_upper"]
            best[L][B] = {"n": int(n), "single": row["single_upper"][0], "vote": row["vote"][n]["upper"][0], "gain": g,
                          "B_over_natural": int(B) / q2[L]["mean_tok_per_draw"]}
            if g[0] < 0: neg.append(-g[0])
            allg += [(x["gain_upper"][0], L, B, k) for k, x in row["vote"].items()]
    sig_pos = [(L, B, n) for L in LAYERS for B, row in q2[L]["B"].items() for n, x in row["vote"].items() if x["gain_upper"][1] > 0]
    worst = min(allg)
    s["10_3"] = {"n_items": {L: q2[L]["n_items"] for L in LAYERS}, "best": best, "best_n_loss_min_pts": 100 * min(neg),
                 "best_n_loss_max_pts": 100 * max(neg), "worst_gain_pts": 100 * worst[0], "worst_cell": list(worst[1:]),
                 "n_cells": sum(len(row["vote"]) for L in LAYERS for row in q2[L]["B"].values()),
                 "positive_ci_cells": [list(x) for x in sig_pos],
                 "natural_tok": {L: q2[L]["mean_tok_per_draw"] for L in LAYERS}}
    # 10.4: operators at own cost, tok ratio = operator tokens / natural single tokens
    ops = {}
    for L in LAYERS:
        v = q3[L]; t = v["single"]["tok"]; ops[L] = {}
        for name, x in [(f"draws_{k}", v["draws"][k]) for k in ("3", "5", "9", "16")] + [(nm, v[nm]) for nm in ("copies_debate", "families_debate") if nm in v]:
            ops[L][name] = {"ratio": x["tok"] / t, "gain": x["gain"]}
    three = ("mmlu-pro", "gpqa-diamond", "gsm-hard")
    sc = [ops[L][k] for L in three for k in ops[L] if k != "families_debate"]
    fam = [ops[L]["families_debate"]["ratio"] for L in three if "families_debate" in ops[L]]
    aux = r["_aux"]
    s["10_4"] = {"ops": ops, "ratio_min": min(x["ratio"] for x in sc), "ratio_max": max(x["ratio"] for x in sc),
                 "gain_min_pts": 100 * min(x["gain"][0] for x in sc), "gain_max_pts": 100 * max(x["gain"][0] for x in sc),
                 "families_ratio_min": min(fam), "families_ratio_max": max(fam),
                 "olymp_single_stopped": aux["olympiad_single_n"] - aux["olympiad_single_finish"].get("length", 0),
                 "olymp_single_n": aux["olympiad_single_n"]}
    # 10.5 / table 10.2
    t2 = {}
    for L in LAYERS:
        v = q4[L]; c = v["comparators"]; h = v["haiku"]; op = c["claude_opus"]["cost_per_item"]
        row = {"opus_acc": c["claude_opus"]["acc"][0], "sonnet_acc": c["claude_sonnet55"]["acc"][0], "haiku1_acc": h["1"]["acc"][0],
               "haiku_over_opus": h["1"]["cost_per_item"] / op, "haiku3_over_opus": h["3"]["cost_per_item"] / op,
               "k_max": max(int(k) for k in h), "opus_think_tok": c["claude_opus"]["think_tok_mean"],
               "opus_think_tok_all_rows": r["_aux"]["opus_think_all_rows"][L]}
        for m, nm in (("claude_opus", "opus"), ("claude_sonnet55", "sonnet")):
            ks = [k for k in sorted(h, key=int) if h[k]["acc"][0] >= c[m]["acc"][0]]
            row[f"match_{nm}"] = None if not ks else {"k": int(ks[0]), "price_ratio": h[ks[0]]["cost_per_item"] / c[m]["cost_per_item"]}
        km = str(row["k_max"]); row["haiku_kmax_acc"] = h[km]["acc"][0]; row["gap_kmax_vs_opus_pts"] = 100 * (row["opus_acc"] - h[km]["acc"][0])
        t2[L] = row
    s["10_5"] = {"table": t2, "haiku_over_opus_min": min(x["haiku_over_opus"] for x in t2.values()),
                 "haiku_over_opus_max": max(x["haiku_over_opus"] for x in t2.values()),
                 "haiku3_over_opus_min": min(x["haiku3_over_opus"] for x in t2.values()),
                 "haiku3_over_opus_max": max(x["haiku3_over_opus"] for x in t2.values()),
                 "opus_think_min": min(x["opus_think_tok"] for x in t2.values()), "opus_think_max": max(x["opus_think_tok"] for x in t2.values()),
                 "opus_think_all_rows_min": min(x["opus_think_tok_all_rows"] for x in t2.values()),
                 "opus_think_all_rows_max": max(x["opus_think_tok_all_rows"] for x in t2.values()),
                 "n_items": {L: q4[L]["n_items"] for L in LAYERS}, "k_available": {L: q4[L]["k_available"] for L in LAYERS}}
    # 10.6
    g9 = {L: {k: q5[L]["gain_pts"]["9"][k] for k in ("0.6", "0.75", "0.9")} for L in LAYERS}
    s["10_6"] = {"phi_min": min(q5[L]["phi"] for L in LAYERS), "phi_max": max(q5[L]["phi"] for L in LAYERS),
                 "rho_min": min(q5[L]["rho_tetra"] for L in LAYERS), "rho_max": max(q5[L]["rho_tetra"] for L in LAYERS),
                 "n9_max_pts": max(v for d in g9.values() for v in d.values()), "n9_olymp_075": g9["olympiad"]["0.75"],
                 "n9_mmlu_min": min(g9["mmlu-pro"].values()), "n9_mmlu_max": max(g9["mmlu-pro"].values()), "n9_gain": g9}
    return s


# ---------- dollar budget (decided 2026-10-19) ----------
POOL = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512",
        "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]
DS = "deepseek_deepseek-v3.2"
DRAW_KS = (1, 3, 5, 9, 16)


def call_cost(pr, model, t_in, t_out):
    """c = pi_in * t_in + pi_out * t_out ($); a missing count (error row) costs 0 on that side."""
    a, b = pr[model]
    return a * (t_in or 0) + b * (t_out or 0)


def best_at(pool, ids, B):
    """best(B) on items `ids`: the most accurate model whose mean natural single-answer cost over `ids` is <= B (ties: cheaper).
    pool = {model: (cost {id: $}, correct {id: 0/1})}. Returns (model, acc, cost) or None when no single answer fits."""
    cand = []
    for m, (c, ok) in pool.items():
        cm = float(np.mean([c[i] for i in ids]))
        if cm <= B: cand.append((float(np.mean([ok[i] for i in ids])), -cm, m))
    if not cand: return None
    acc, mc, m = max(cand)
    return m, acc, -mc


def _vs(boot, acc, ref, ids):
    return boot(np.array([acc[i] for i in ids], float) - np.array([ref[i] for i in ids], float))


def compute_dollar(boot=None):
    """Dollar-budget versions of v8 §10.3 (table 10.1) and §10.4 (decided 2026-10-19). Data: P2_DATA (default uncapped).
    Prices: datasets/model-prices (OpenRouter models: billed provider mix; Claude: LIST prices, not billed).
    Pool for best(B): paper 1's uncapped 7-model corpus (natural T=0 answer of each model, label audits; a 65,536 hit = wrong). Cost of a
      model's answer on an item = pi_in * t_in + pi_out * t_out, t_in from the tn5 T=0 matrix row of the same (item, model) (same
      prompt; the model's layer median where the matrix has no row), t_out = the corpus answer's out_tok.
    best(C) is chosen in-sample on the operator's items (max over 7 accuracies: biased up, so gains against it are conservative) and held
      fixed in the bootstrap; gain = mean over items of acc_i(A) - correct_i(best); 2000 item-bootstrap draws from ONE numpy
      Generator(seed 0) consumed in the order t10_1 (layer, B, n: vote, gain vs single, gain vs best) -> ops (layer, operator).
    t10_1 (table 10.1 in dollars): one DeepSeek T=1 draw cut at B tokens (mean over its 16 draws) costs c1 = mean pi_in*t_in +
      pi_out*min(t_out, B); n draws are each cut at b = the largest cut of the stored grid with mean cost of draws 0..n-1 <= c1 (v8:
      b = B/n). Votes and the upper bracket as Q2 (capped cut traces: cut-correct.jsonl has no uncapped twin).
    ops (§10.4): k natural T=1 DeepSeek draws (plurality, paper 1 rule); copies debate (6 DeepSeek, arm plain) and families debate
      (6 models) INCLUDING round 0 (copies: draws 0..5; families: the six models' T=0 matrix answers), rounds 1-3 priced per call from
      the tn5 debate rows (tok_in carries the peers' answers); v8's token ratio counted output tokens of rounds 1-3 only.
    same_model: dollar ratio / output-token ratio of k draws vs the one natural T=0 answer of the same model."""
    boot = boot or Boot(0)
    pr = IO.prices(); mt = IO.matrix_tokens(); st = IO.sample_tokens(DS)
    cut = IO.cut_rows(); nat = IO.natural(); deb = IO.debates()
    dtok = {k: IO.debate_tokens(k) for k in ("copies", "families")}
    grid = sorted({int(b) for b in cut[0]["cut"]})
    out = {"prices_per_M": {m.replace(".", "_"): [pr[m][0] * 1e6, pr[m][1] * 1e6] for m in POOL},   # no dots: output_ref paths
           "t10_1": {}, "ops": {}, "same_model": {}}
    pools, pids = {}, {}
    for L in LAYERS:
        P = IO.pool_layer(L)
        med = {m: float(np.median([v[0] for (i, mm), v in mt.items() if mm == m and i in P])) for m in POOL}
        pools[L] = {m: ({i: call_cost(pr, m, mt.get((i, m), (med[m], None))[0], P[i][m]["out_tok"]) for i in P},
                        {i: float(P[i][m]["correct"]) for i in P}) for m in POOL}
        pids[L] = set(P)
    # --- table 10.1 in dollars ---
    for L in LAYERS:
        by = collections.defaultdict(dict)
        for r in cut:
            if r["layer"] == L and r["stage"] == "samples": by[r["id"]][r["sample"]] = r
        ids = sorted(i for i, d in by.items() if len(d) >= 16)
        tin = {(i, s): (st.get((i, s)) or (None, None))[0] for i in ids for s in range(16)}
        tin_med = float(np.median([v for v in tin.values() if v is not None]))
        tin = {k: (v if v is not None else tin_med) for k, v in tin.items()}

        def cost_cut(n, b):
            return float(np.mean([sum(call_cost(pr, DS, tin[(i, s)], min(by[i][s]["out_tok"], b)) for s in range(n)) for i in ids]))
        ids_pool = [i for i in ids if i in pids[L]]
        res = {"n_items": len(ids), "n_items_pool": len(ids_pool), "mean_tin": float(np.mean(list(tin.values()))), "B": {}}
        for B in Q2B:
            c1 = float(np.mean([np.mean([call_cost(pr, DS, tin[(i, s)], min(by[i][s]["out_tok"], B)) for s in range(16)]) for i in ids]))
            sing = {i: float(np.mean([by[i][s]["cut"][str(B)]["upper"] for s in range(16)])) for i in ids}
            row = {"cost_single": c1, "single_upper": boot([sing[i] for i in ids]), "vote": {}}
            for n in NS[1:]:
                fit = [b for b in grid if cost_cut(n, b) <= c1]
                if not fit:
                    row["vote"][str(n)] = None; continue
                b = max(fit)
                vu = {i: float(right(plurality([by[i][s]["cut"][str(b)]["vote"] for s in range(n)]))) for i in ids}
                cell = {"b": b, "b_token_equal": B // n, "cost": cost_cut(n, b), "upper": boot([vu[i] for i in ids]),
                        "gain_vs_single": _vs(boot, vu, sing, ids)}
                bb = best_at(pools[L], ids_pool, cell["cost"])
                cell["best"] = None
                if bb:
                    cell["best"] = {"model": bb[0], "acc": bb[1], "cost": bb[2]}
                    cell["gain_vs_best"] = _vs(boot, vu, pools[L][bb[0]][1], ids_pool)
                row["vote"][str(n)] = cell
            res["B"][str(B)] = row
        out["t10_1"][L] = res
    # --- §10.4 operators at their own dollar cost ---
    for L in LAYERS:
        nby = collections.defaultdict(dict)
        for r in nat:
            if r["layer"] == L and r["stage"] == "samples" and r["out_tok"]: nby[r["id"]][r["sample"]] = r
        ids = sorted(i for i in nby if len(nby[i]) >= 16 and i in pids[L])
        tin_med = float(np.median([v[0] for (i, s), v in st.items() if i in nby]))

        def tin_of(i, s): return (st.get((i, s)) or (tin_med,))[0]

        def dcost(i, s): return call_cost(pr, DS, tin_of(i, s), nby[i][s]["out_tok"])
        ops = {}
        for k in DRAW_KS:
            ops[f"draws_{k}"] = ({i: vote_right([nby[i][s] for s in range(k)]) for i in ids},
                                 {i: sum(dcost(i, s) for s in range(k)) for i in ids},
                                 {i: sum(nby[i][s]["out_tok"] for s in range(k)) for i in ids}, {i: 0.0 for i in ids})
        for kind, nm in (("copies", "copies_debate"), ("families", "families_debate")):
            src = collections.defaultdict(list)
            for r in deb:
                if r["layer"] == L and r["kind"] == kind and r["id"] in pids[L]: src[r["id"]].append(r)
            okids = [i for i in ids if src[i]]
            if not okids: continue
            acc, cst, tok, c0 = {}, {}, {}, {}
            for i in okids:
                rs = src[i]; last = max(r["round"] for r in rs)
                acc[i] = vote_right([r for r in rs if r["round"] == last and r["finish"] != "length"])
                c = 0.0; t = 0
                for r in rs:
                    dflt = (DS if kind == "copies" else IO.SERVED[r["agent"]], None, r["out_tok"])
                    m, ti, to = dtok[kind].get((i, r["round"], r["agent"]), dflt)
                    c += call_cost(pr, m, ti, to if to is not None else r["out_tok"]); t += r["out_tok"] or 0
                if kind == "copies":   # round 0 = DeepSeek T=1 draws 0..5
                    r0 = sum(dcost(i, s) for s in range(6) if s in nby[i])
                else:                  # round 0 = the six models' T=0 matrix answers
                    r0 = sum(call_cost(pr, m, *mt[(i, m)]) for m in POOL if m != "claude_opus" and (i, m) in mt)
                cst[i] = c + r0; tok[i] = t; c0[i] = r0
            ops[nm] = (acc, cst, tok, c0)
        res = {}
        ds_c, ds_ok = pools[L][DS]
        for nm, (acc, cst, tok, c0) in ops.items():
            I = sorted(acc)
            C = float(np.mean([cst[i] for i in I])); c_ds = float(np.mean([ds_c[i] for i in I]))
            bb = best_at(pools[L], I, C)
            row = {"n": len(I), "cost": C, "cost_round0": float(np.mean([c0[i] for i in I])), "acc": boot([acc[i] for i in I]),
                   "ds_single_cost": c_ds, "ds_single_acc": float(np.mean([ds_ok[i] for i in I])), "dollar_ratio": C / c_ds,
                   "out_tok": float(np.mean([tok[i] for i in I]))}
            row["best"] = None
            if bb:
                row["best"] = {"model": bb[0], "acc": bb[1], "cost": bb[2]}
                row["gain_vs_best"] = _vs(boot, acc, pools[L][bb[0]][1], I)
            row["gain_vs_ds_single"] = _vs(boot, acc, ds_ok, I)
            res[nm] = row
        out["ops"][L] = res
        # same model: k draws vs its one natural T=0 answer, dollars vs output tokens
        P = IO.pool_layer(L)
        t1 = float(np.mean([P[i][DS]["out_tok"] or 0 for i in ids]))
        tin_mean = float(np.mean([tin_of(i, s) for i in ids for s in range(16)]))
        out_mean = float(np.mean([nby[i][s]["out_tok"] for i in ids for s in range(16)]))
        a, b = pr[DS]
        sm = {"mean_tin": tin_mean, "mean_tout_draw": out_mean, "mean_tout_t0": t1,
              "input_share_of_draw_cost": a * tin_mean / (a * tin_mean + b * out_mean), "k": {}}
        for k in DRAW_KS:
            x = res[f"draws_{k}"]; tr = x["out_tok"] / t1
            sm["k"][str(k)] = {"token_ratio": tr, "dollar_ratio": x["dollar_ratio"], "dollar_over_token": x["dollar_ratio"] / tr}
        out["same_model"][L] = sm
    cells = [(c, L, B, n) for L, v in out["t10_1"].items() for B, row in v["B"].items() for n, c in row["vote"].items() if c]
    gs = max((c["gain_vs_single"][0], L, B, n) for c, L, B, n in cells)
    gb = max((c["gain_vs_best"][0], L, B, n) for c, L, B, n in cells if "gain_vs_best" in c)
    out["summary"] = {
        "t10_1_cells": len(cells),
        "t10_1_vs_single_max_pts": 100 * gs[0], "t10_1_vs_single_max_cell": list(gs[1:]),
        "t10_1_vs_single_pos_ci": [[L, B, n] for c, L, B, n in cells if c["gain_vs_single"][1] > 0],
        "t10_1_vs_best_max_pts": 100 * gb[0], "t10_1_vs_best_max_cell": list(gb[1:]),
        "t10_1_vs_best_pos_ci": [[L, B, n] for c, L, B, n in cells if "gain_vs_best" in c and c["gain_vs_best"][1] > 0],
        "ops_vs_best_pos_ci": [[L, k] for L, v in out["ops"].items() for k, x in v.items() if x.get("gain_vs_best") and x["gain_vs_best"][1] > 0],
        "ops_vs_best_neg_ci": [[L, k] for L, v in out["ops"].items() for k, x in v.items() if x.get("gain_vs_best") and x["gain_vs_best"][2] < 0]}
    return out


# Unit of the §10 budget comparisons: "usd" (default, decision 2026-10-19) adds the dollar tables (key "dollar"); "tokens" = v8 only
# (the equal-token Q1–Q5 tables; tests pin it for the v8 regressions). The v8 keys q1..q5 / s10 are computed in both modes.
UNIT = os.environ.get("P2_BUDGET_UNIT", "usd")
assert UNIT in ("usd", "tokens"), UNIT


def run() -> dict:
    r = compute_q()
    sec = section10(r)
    aux = r.pop("_aux"); aux = {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in aux.items()}
    res = {**r, "aux": aux, "s10": sec, "untraced": untraced(r["q5"])}
    if UNIT == "usd": res["dollar"] = compute_dollar()
    return {"budget": res}


if __name__ == "__main__":
    import json, sys
    out = run()["budget"]
    json.dump(out, sys.stdout, indent=1, default=float)
