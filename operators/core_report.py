"""Paper 2 on EXISTING data (decided 2026-10-20): every generative operator on the core of the paper-1 corpus
(the 236 items of datasets/set_a/tn5-samples = first 60 per layer; 227 of them in the corpus after the label audit; §1), uncapped rerun
data by default (P2_DATA=uncapped, p2parts/datasrc.py), paper-1 answer rule (p2parts/answer_id.py, RULE p1), plurality tie
rule 'first' (fixed participant order). Two questions per operator: (1) does it add anything over its own starting point,
cost aside; (2) if so, what it costs in dollars (datasets/model-prices, p2parts/budget.call_cost).

Operators A and starting points A0 (same items):
- sc.<model>.k5 / k16: plurality of T=1 draws 0..k-1 (DeepSeek V3.2, Haiku 4.5, GLM 5.3 Flash; tn5-samples) vs one draw.
  A0 = draw 0 (decided 2026-10-20): acc0, Δ, its CI, W / L, McNemar, s(A), C(A0) and $/extra item are all paired
  against the same single answer. "mean_draw" (reference only) = mean single-draw accuracy over all stored positions (16;
  Haiku GPQA / olympiad: 8). k = 16 only where 16 draws exist (Haiku: MMLU-Pro, GSM-Hard). Items whose draw 0 is unanswered are dropped. W / L / McNemar are
  (draw 0). Participants = draws 0..k-1.
- debate.mixed: six-model debate (tn5-debate-mixed, round 0 = tn5-matrix T=0 answers), round-3 plurality vs round-0
  plurality of the same six. Participants = the six round-0 answers.
- debate.copies.<arm>: six DeepSeek copies (tn5-debate-homo; arms plain, selfonly, pressure), round-3 plurality vs round-0
  plurality (= draws 0..5). Participants = the six round-0 draws.
- synth: one DeepSeek call over the six mixed-debate round-3 answers (tn5-aggregate mode synth) vs plurality of those inputs;
  moa: one DeepSeek call over five of its own T=1 draws (mode moa) vs plurality of the five. Participants = the inputs.
Missing answers (correct None) are missing votes; a missing participant counts as wrong in best / oracle. A 65,536 hit is wrong.

Metrics per operator x layer and pooled over the core:
- acc(A), acc(A0), Δ with paired item bootstrap 95% percentile CI (10,000 resamples, numpy Generator seed 0, consumed in the
  fixed order operator -> row -> [Δ, gain vs best(C)]); exact McNemar p on W vs L.
- W (A0 wrong, A right), L (A0 right, A wrong), C⁺ (all participants wrong, A right). For SC and for plurality operators C⁺ = 0
  by construction (a plurality returns a participant's answer).
- best over participants: in-sample (max accuracy) and held-out (2-fold x 20 splits over items, stratified by layer, seed 0:
  choose on one half, score on the other; per-item value averaged over splits); oracle; prize = oracle - best_heldout;
  s(A) = (acc(A) - best)/(oracle - best) (held-out and in-sample); flagged "prize < 2 pts" when the prize is under 0.02.
- cost: C(A) and C(A0) = mean $ per item over all calls including round 0 and input tokens (call_cost = pi_in*t_in +
  pi_out*t_out, missing counts = 0). best(C(A)) = most accurate natural T=0 answer of the 7-model pool (paper-1 corpus,
  cost as in budget.compute_dollar) with mean cost <= C(A): held-out (same splits; cost and accuracy measured on the choice
  half) and in-sample; gain vs best(C) with bootstrap CI (choice fixed); $ per extra solved item = (C(A) - C(A0)) / Δ, Δ > 0.
- saturation: prize of the 7-model T=0 pool on the same items (in-sample best; = summaries/out2/core_vs_rest P7 core).
Checks on OTHER item sets (block "checks", not the core): weak pool W debate / synthesis (debx-weak-pool, its own 196 / 156
items, round 0 from the capped matrix) and decomposition (decomposition.json, 8,192 protocol, 97 selected core items).
Offline, $0. Usage: python3 operators/core_report.py  -> summaries/out2/core_operators.{json,md}
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2parts import tn5 as IO  # noqa: E402
from p2parts import budget as BIO  # noqa: E402
from p2parts import debx as DX  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAYERS = IO.HARD
SC_MODELS = {"deepseek": "deepseek_deepseek-v3.2", "haiku": "claude_haiku", "glm": "z-ai_glm-5.3-flash"}
POOL = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512",
        "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]
DS = "deepseek_deepseek-v3.2"
NBOOT, NSPLIT, SMALL_PRIZE = 10000, 20, 0.02


# ---------------- statistics ----------------
class Boot:
    def __init__(self, seed=0): self.g = np.random.default_rng(seed)

    def ci(self, d):
        d = np.asarray(d, float); n = len(d)
        m = self.g.integers(0, n, size=(NBOOT, n))
        b = d[m].mean(1)
        return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def splits(layers, seed=0):
    """NSPLIT 2-fold splits stratified by layer: list of boolean masks (True = half 0)."""
    g = np.random.default_rng(seed); layers = np.asarray(layers); out = []
    for _ in range(NSPLIT):
        m = np.zeros(len(layers), bool)
        for L in sorted(set(layers)):
            idx = np.flatnonzero(layers == L); g.shuffle(idx)
            m[idx[: len(idx) // 2]] = True
        out.append(m)
    return out


def heldout_best(P, S):
    """P: items x participants 0/1. Per-item held-out value of the participant chosen on the other half, averaged over S."""
    v = np.zeros(P.shape[0]); picks = collections.Counter()
    for m in S:
        for a, b in ((m, ~m), (~m, m)):
            j = int(np.argmax(P[a].mean(0)))
            v[b] += P[b, j]; picks[j] += 1
    return v / len(S), picks


def heldout_bestC(acc, cost, CA, S, names):
    """acc, cost: {model: per-item array}; CA: per-item operator cost. Per-item held-out value of best(C(A))."""
    v = np.zeros(len(CA)); picks = collections.Counter(); none = 0
    for m in S:
        for a, b in ((m, ~m), (~m, m)):
            B = CA[a].mean()
            cand = [(acc[k][a].mean(), -cost[k][a].mean(), k) for k in names if cost[k][a].mean() <= B]
            if not cand:
                none += 1; continue
            k = max(cand)[2]
            v[b] += acc[k][b]; picks[k] += 1
    if none:
        return None, picks, none
    return v / len(S), picks, none


def insample_bestC(acc, cost, CA, names):
    B = CA.mean()
    cand = [(acc[k].mean(), -cost[k].mean(), k) for k in names if cost[k].mean() <= B]
    return max(cand)[2] if cand else None


# ---------------- data ----------------
def call_cost(pr, model, t_in, t_out):
    """Same as operators/budget.call_cost: pi_in * t_in + pi_out * t_out ($); a missing count costs 0 on that side."""
    a, b = pr[model]
    return a * (t_in or 0) + b * (t_out or 0)


def core_ids():
    """tn5-samples items (236) that are also in paper 1's corpus (label audits wave 1+2): 227 = 58/58/57/54, the core of
    summaries/core_vs_rest.py. The 9 dropped (gpqa:19, gpqa:38, mmlupro:2927, aime24:21/22/26, aime25:3/7/9) are not in the corpus."""
    tn = {(r["layer"], r["id"]) for m in SC_MODELS.values() for L in LAYERS for r in IO.samples(m, L)}
    corp = {(L, i) for L in LAYERS for i in BIO.pool_layer(L)}
    return sorted(tn & corp)


def pool_tables():
    """{(layer,id): {model: (correct, $)}} for the 7-model natural T=0 answers (as budget.compute_dollar)."""
    pr = BIO.prices(); mt = BIO.matrix_tokens(); out = {}
    for L in LAYERS:
        P = BIO.pool_layer(L)
        med = {m: float(np.median([v[0] for (i, mm), v in mt.items() if mm == m and i in P and v[0] is not None])) for m in POOL}
        for i, d in P.items():
            out[(L, i)] = {m: (float(bool(d[m]["correct"])),
                               call_cost(pr, m, (mt.get((i, m)) or (med[m], None))[0] or med[m], d[m]["out_tok"])) for m in POOL}
    return out


def cost_row(pr, model, r):
    return call_cost(pr, model, r["tok"]["in"], r["tok"]["out"])


def build_ops():
    """{op: {"items": [(layer,id)], "a": [], "a0": [], "w0": [] (binary baseline for W/L), "P": [[...]], "c": [], "c0": []}}."""
    pr = BIO.prices(); core = set(core_ids()); ops = {}
    # ---- self-consistency ----
    draws = {}
    for short, m in SC_MODELS.items():
        for L in LAYERS:
            for r in IO.samples(m, L):
                draws.setdefault((short, L, r["id"]), {})[r["sample"]] = r
    for short, m in SC_MODELS.items():
        for k in (5, 16):
            o = ops[f"sc_{short}_k{k}"] = collections.defaultdict(list)
            for (L, i) in sorted(core):
                d = draws.get((short, L, i))
                if not d or not all(s in d for s in range(k)): continue      # Haiku: 8 draws on GPQA / olympiad -> no k=16 there
                pres = [r for r in d.values() if r.get("correct") is not None]
                first = [d[s] for s in range(k) if d[s].get("correct") is not None]
                if not first or d[0].get("correct") is None: continue
                o["items"].append((L, i))
                o["a"].append(float(IO.maj_correct(first)))
                o["md"].append(float(np.mean([bool(r["correct"]) for r in pres])))   # reference column "mean draw"
                o["a0"].append(float(bool(d[0]["correct"])))                          # A0 = draw 0 (2026-10-20)
                o["w0"].append(float(bool(d[0]["correct"])))
                o["P"].append([float(bool(d[s].get("correct"))) for s in range(k)])
                o["c"].append(sum(cost_row(pr, m, d[s]) for s in range(k)))
                o["c0"].append(cost_row(pr, m, d[0]))
    # ---- mixed debate ----
    deb = IO.load_debate()
    mixed_cost = {}
    o = ops["debate_mixed"] = collections.defaultdict(list)
    for i, d in sorted(deb.items()):
        L = next(iter(d.values()))["layer"]
        if (L, i) not in core: continue
        r0 = [d.get((a, 0)) for a in IO.DEBATE_AGENTS]; r3 = [d.get((a, 3)) for a in IO.DEBATE_AGENTS]
        if any(x is None for x in r0) or not any(x for x in r3): continue
        c0 = sum(cost_row(pr, BIO.SERVED[a], d[(a, 0)]) for a in IO.DEBATE_AGENTS)
        c = c0 + sum(cost_row(pr, BIO.SERVED[a], d[(a, rd)]) for a in IO.DEBATE_AGENTS for rd in (1, 2, 3) if (a, rd) in d)
        mixed_cost[(L, i)] = c
        o["items"].append((L, i))
        o["a"].append(float(IO.maj_correct([x for x in r3 if x])))
        o["a0"].append(float(IO.maj_correct(r0))); o["w0"].append(o["a0"][-1])
        o["P"].append([float(bool(x.get("correct"))) for x in r0])
        o["c"].append(c); o["c0"].append(c0)
    # ---- copies debate ----
    for arm in IO.ARMS:
        hd = IO.load_homo(arm)
        o = ops[f"debate_copies_{arm}"] = collections.defaultdict(list)
        for i, d in sorted(hd.items()):
            L = next((r["layer"] for r in d.values() if "layer" in r), None)
            if (L, i) not in core: continue
            ag = [f"deepseek/deepseek-v3.2#{s}" for s in range(6)]
            r0 = [d.get((a, 0)) for a in ag]; r3 = [d.get((a, 3)) for a in ag]
            if any(x is None for x in r0) or not any(x for x in r3): continue
            c0 = sum(cost_row(pr, DS, x) for x in r0)
            c = c0 + sum(cost_row(pr, DS, d[(a, rd)]) for a in ag for rd in (1, 2, 3) if (a, rd) in d)
            o["items"].append((L, i))
            o["a"].append(float(IO.maj_correct([x for x in r3 if x])))
            o["a0"].append(float(IO.maj_correct(r0))); o["w0"].append(o["a0"][-1])
            o["P"].append([float(bool(x.get("correct"))) for x in r0])
            o["c"].append(c); o["c0"].append(c0)
    # ---- synthesis / MoA ----
    for mode in ("synth", "moa"):
        o = ops[mode] = collections.defaultdict(list)
        for L in LAYERS:
            for r in IO.aggregate(mode, L):
                i = r["id"]
                if (L, i) not in core or r.get("correct") is None: continue
                ins = r["inputs"]
                call = call_cost(pr, DS, r["tok"]["in"], r["tok"]["out"])
                if mode == "synth":
                    if (L, i) not in mixed_cost: continue
                    c0 = mixed_cost[(L, i)]      # the inputs = the whole mixed debate (round 0..3)
                else:
                    d = draws.get(("deepseek", L, i))
                    if not d: continue
                    c0 = sum(cost_row(pr, DS, d[s]) for s in range(5) if s in d)   # stored inputs: five draws (ids not kept)
                o["items"].append((L, i))
                o["a"].append(float(bool(r["correct"])))
                o["a0"].append(float(IO.maj_correct([x for x in ins if x.get("correct") is not None])))
                o["w0"].append(o["a0"][-1])
                o["P"].append([float(bool(x.get("correct"))) for x in ins])
                o["c"].append(c0 + call); o["c0"].append(c0)
    return ops


# ---------------- report ----------------
def row_stats(boot, items, a, a0, w0, P, c, c0, pool, md=None):
    n = len(items); a, a0, w0, c, c0 = map(np.asarray, (a, a0, w0, c, c0))
    width = max(len(p) for p in P); P = np.array([p + [0.0] * (width - len(p)) for p in P])
    lay = [L for L, _ in items]; S = splits(lay)
    d = boot.ci(a - a0)
    W = int(((w0 == 0) & (a == 1)).sum()); Lo = int(((w0 == 1) & (a == 0)).sum())
    allw = P.max(1) == 0
    Cp = int((allw & (a == 1)).sum())
    oracle = float(P.max(1).mean()); best_in = float(P.mean(0).max())
    hb, _ = heldout_best(P, S); best_ho = float(hb.mean())
    prize = oracle - best_ho
    sh = lambda b: (float((a.mean() - b) / (oracle - b)) if oracle - b > 1e-12 else None)
    # 7-model pool on the same items
    acc = {m: np.array([pool[x][m][0] for x in items]) for m in POOL}
    cost = {m: np.array([pool[x][m][1] for x in items]) for m in POOL}
    sat = float(np.max(np.stack([acc[m] for m in POOL]), 0).mean() - max(acc[m].mean() for m in POOL))
    v, picks, none = heldout_bestC(acc, cost, c, S, POOL)
    bc_in = insample_bestC(acc, cost, c, POOL)
    out = {"N": n, "acc0": float(a0.mean()), "mean_draw": (float(np.mean(md)) if md else None), "acc": float(a.mean()), "delta": d, "W": W, "L": Lo,
           "mcnemar_p": IO.mcnemar(W, Lo), "Cplus": Cp, "all_wrong": int(allw.sum()), "oracle": oracle,
           "best_insample": best_in, "best_heldout": best_ho, "prize": prize, "prize_insample": oracle - best_in,
           "s_heldout": sh(best_ho), "s_insample": sh(best_in), "prize_too_small": prize < SMALL_PRIZE,
           "C0_usd": float(c0.mean()), "C_usd": float(c.mean()), "pool_prize_saturation": sat,
           # resampling operators (SC, copies): headroom = pass@k (= oracle over the draws) - mean participant accuracy;
           # share = (acc - mean participant) / headroom (2026-10-26; replaces prize/s(A), whose "best draw" is noise)
           "mean_participant": float(P.mean()),
           "resample_share": (float((a.mean() - P.mean()) / (oracle - P.mean())) if oracle - P.mean() > 1e-12 else None),
           "usd_per_extra_item": (float((c.mean() - c0.mean()) / d[0]) if d[0] > 0 else None)}
    if v is None:
        out["bestC_heldout"] = None; out["gain_vs_bestC"] = None
    else:
        out["bestC_heldout"] = {"acc": float(v.mean()), "picks": dict(picks)}
        out["gain_vs_bestC"] = boot.ci(a - v)
    if bc_in:
        out["bestC_insample"] = {"model": bc_in, "acc": float(acc[bc_in].mean()), "cost": float(cost[bc_in].mean()),
                                 "gain": float(a.mean() - acc[bc_in].mean())}
    else:
        out["bestC_insample"] = None
    if v is None:  # keep the bootstrap stream order fixed
        boot.ci(a - a0)
    return out


def checks(boot):
    """Weak pool W (debx-weak-pool) and decomposition: other item sets, not the core."""
    r0 = collections.defaultdict(dict)
    for r in DX.weak_round0(): r0[r["id"]][r["agent"]] = r
    ags = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "claude:haiku"]
    last = {}
    for r in DX.weak_debate(): last[(r["id"], r["round"], r["agent"])] = r     # last row per key wins
    ids = sorted({k[0] for k in last})
    res = {}
    a, a0, P = [], [], []
    for i in ids:
        if not all(g in r0[i] for g in ags): continue
        r3 = [last.get((i, 3, g)) for g in ags]
        if not any(r3): continue
        a.append(float(IO.maj_correct([x for x in r3 if x and x.get("correct") is not None])))
        a0.append(float(IO.maj_correct([r0[i][g] for g in ags])))
        P.append([float(bool(r0[i][g].get("correct"))) for g in ags])
    a, a0, P = map(np.asarray, (a, a0, P))
    W = int(((a0 == 0) & (a == 1)).sum()); Lo = int(((a0 == 1) & (a == 0)).sum())
    res["weak_debate"] = {"items": "debx-weak-pool debate-w (196 = contested + all-wrong + control of the 733 W items)",
                          "N": len(a), "acc0": float(a0.mean()), "acc": float(a.mean()), "delta": boot.ci(a - a0), "W": W, "L": Lo,
                          "mcnemar_p": IO.mcnemar(W, Lo), "Cplus": int(((P.max(1) == 0) & (a == 1)).sum()),
                          "all_wrong": int((P.max(1) == 0).sum()), "oracle": float(P.max(1).mean()),
                          "best_insample": float(P.mean(0).max())}
    a, a0, aw = [], [], []
    for r in DX.weak_synth():
        i = r["id"]
        if r.get("correct") is None or not all(g in r0[i] for g in ags): continue
        a.append(float(bool(r["correct"]))); a0.append(float(IO.maj_correct([r0[i][g] for g in ags])))
        aw.append(not any(r0[i][g].get("correct") for g in ags))
    a, a0, aw = map(np.asarray, (a, a0, aw))
    W = int(((a0 == 0) & (a == 1)).sum()); Lo = int(((a0 == 1) & (a == 0)).sum())
    res["weak_synth"] = {"items": "debx-weak-pool synth-w (contested + all-wrong; inputs = round 0 of W, capped 8,192)",
                         "N": len(a), "acc0": float(a0.mean()), "acc": float(a.mean()), "delta": boot.ci(a - a0), "W": W, "L": Lo,
                         "mcnemar_p": IO.mcnemar(W, Lo), "Cplus": int((aw & (a == 1)).sum()), "all_wrong": int(aw.sum())}
    dj = json.load(open(ROOT / "summaries" / "out2" / "decomposition.json"))
    t = dj["table"]; u = dj["untraced"]
    res["decomposition"] = {
        "items": "decomposition selection.json (97 core items, strata by base T=1 16-draw count), 8,192-token protocol",
        "deepseek_hard_true_delta_p4_numeric": u["ds_hard_true"]["delta_p4_numeric"],
        "deepseek_hard_true_delta_p4_numeric_ci": u["ds_hard_true"]["delta_p4_numeric_ci"],
        "deepseek_hard_true_zero16_numeric": u["ds_hard_true"]["zero16_numeric"],
        "deepseek_medium_easy_delta": -u["medium_easy"]["deepseek"]["loss"],   # stored as a loss (positive)
        "deepseek_medium_easy_ci": u["medium_easy"]["deepseek"]["delta_ci"],
        "deepseek_hard_true": {k: t["deepseek"]["hard-true"][k] for k in ("n", "single_p4", "arm2_p4", "arm3_p4", "zero16", "zero16_solved")},
        "glm_hard_cap": {k: t["glm"]["hard-cap"][k] for k in ("n", "single_p4", "arm2_p4", "arm3_p4", "zero16", "zero16_solved")},
        "source": "summaries/out2/decomposition.json (operators/decomposition.py)"}
    return res


ORDER = ["sc_deepseek_k5", "sc_haiku_k5", "sc_glm_k5", "sc_deepseek_k16", "sc_haiku_k16", "sc_glm_k16", "debate_mixed",
         "debate_copies_plain", "debate_copies_selfonly", "debate_copies_pressure", "synth", "moa"]


def compute():
    boot = Boot(0); pool = pool_tables(); ops = build_ops(); out = {"ops": {}}
    for name in ORDER:
        o = ops[name]; rows = {}
        for L in LAYERS + ["pooled"]:
            sel = [j for j, x in enumerate(o["items"]) if L == "pooled" or x[0] == L]
            if len(sel) < 10: continue          # e.g. Haiku k=16 on olympiad: 2 items with 16 draws
            g = lambda k: [o[k][j] for j in sel]
            rows[L] = row_stats(boot, g("items"), g("a"), g("a0"), g("w0"), g("P"), g("c"), g("c0"), pool, g("md") if o.get("md") else None)
        out["ops"][name] = rows
    out["checks"] = checks(boot)
    out["method"] = {"nboot": NBOOT, "nsplit": NSPLIT, "seed": 0, "data": __import__("p2parts.datasrc", fromlist=["x"]).DATA,
                     "answer_rule": IO.AID.RULE, "core_items": len(core_ids())}
    return out


def md(R):
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    pt = lambda x: "—" if x is None else f"{100 * x:+.1f}"
    ci = lambda c: "—" if c is None else f"{pt(c[0])} [{pt(c[1])}, {pt(c[2])}]"
    usd = lambda x: "—" if x is None else (f"{x:.4f}" if x < 1 else f"{x:.2f}")
    L = ["# Paper 2 operators on the core (existing data)", "",
         f"Data: P2_DATA={R['method']['data']}, answer rule {R['method']['answer_rule']}, core {R['method']['core_items']} items; "
         f"bootstrap {R['method']['nboot']} (seed 0), held-out best = 2-fold x {R['method']['nsplit']} splits. Points (pts) for Δ and gains; "
         "prize/s use held-out best over the operator's participants; sat = prize of the 7-model T=0 pool on the same items. "
         "* = prize < 2 pts (s not measurable). SC: A0 = draw 0 for every statistic; 'mean draw' = reference. Generated by operators/core_report.py.", ""]
    hdr = ("| set | N | acc0 | mean draw | acc | Δ pts [CI] | p | W | L | C⁺ | prize | s | sat | C(A0)$ | C(A)$ | vs best(C) pts [CI] | $/extra |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        L += [f"## {name}", "", *hdr]
        for k, s in R["ops"][name].items():
            L.append(f"| {k} | {s['N']} | {f(s['acc0'])} | {f(s.get('mean_draw'))} | {f(s['acc'])} | {ci(s['delta'])} | {f(s['mcnemar_p'], 3)} | {s['W']} | {s['L']} | "
                     f"{s['Cplus']} | {100 * s['prize']:.1f}{'*' if s['prize_too_small'] else ''} | {f(s['s_heldout'], 2)} | "
                     f"{100 * s['pool_prize_saturation']:.1f} | {usd(s['C0_usd'])} | {usd(s['C_usd'])} | {ci(s['gain_vs_bestC'])} | "
                     f"{usd(s['usd_per_extra_item'])} |")
        b = R["ops"][name].get("pooled", {}).get("bestC_insample")
        if b: L.append(f"\nPooled in-sample best(C): {b['model']} acc {b['acc']:.3f} at ${b['cost']:.4f} (gain {pt(b['gain'])} pts).")
        L.append("")
    c = R["checks"]
    L += ["## CHECKS on other item sets (not the core)", ""]
    for k in ("weak_debate", "weak_synth"):
        s = c[k]
        L.append(f"- **{k}** ({s['items']}): N {s['N']}, acc0 {s['acc0']:.3f} → acc {s['acc']:.3f}, Δ {ci(s['delta'])} pts, "
                 f"W {s['W']} / L {s['L']}, p {s['mcnemar_p']:.4f}, C⁺ {s['Cplus']} of {s['all_wrong']} all-wrong.")
    d = c["decomposition"]
    L.append(f"- **decomposition** ({d['items']}): DeepSeek hard-true numeric Δ pass@4 {d['deepseek_hard_true_delta_p4_numeric']:+.2f} "
             f"[{d['deepseek_hard_true_delta_p4_numeric_ci'][0]:+.2f}, {d['deepseek_hard_true_delta_p4_numeric_ci'][1]:+.2f}], "
             f"never-solved numeric {d['deepseek_hard_true_zero16_numeric']}; DeepSeek medium/easy Δ {d['deepseek_medium_easy_delta']:+.2f} "
             f"[{d['deepseek_medium_easy_ci'][0]:+.2f}, {d['deepseek_medium_easy_ci'][1]:+.2f}]; GLM hard-cap single p4 "
             f"{d['glm_hard_cap']['single_p4']:.2f} → split {d['glm_hard_cap']['arm2_p4']:.2f} (gain only where the 8,192 cap bit). "
             f"Source {d['source']}.")
    return "\n".join(L) + "\n"


def run() -> dict:
    R = compute()
    return {"core_operators": R, "core_operators.md": md(R)}


if __name__ == "__main__":
    res = run()
    out = ROOT / "summaries" / "out2"
    (out / "core_operators.json").write_text(json.dumps(res["core_operators"], indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "core_operators.md").write_text(res["core_operators.md"])
    print(res["core_operators.md"])
