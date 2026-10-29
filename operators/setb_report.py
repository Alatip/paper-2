"""Paper 2, set B (confirmatory experiment): every generative operator vs its starting point. Reads only public data:
datasets/set_b/{items,answers}.csv (text-free; built from the private runs). Offline, $0.

    python3 operators/setb_report.py                    -> summaries/out2/setB.{json,md}
    SETB_NO_DEFECTIVE=1 python3 operators/setb_report.py -> summaries/out2/setB_nodefective.{json,md} (sensitivity)

Correctness per answer is stored (screening grader + gold audit). Answer identity = opaque code (paper-1 rule, groups
chained over all answers of an item); ties 'first' in the fixed pool order (Haiku, Sonnet, Opus, DeepSeek, GLM, Qwen, Kimi).

Operators (A) vs starting points (A0), same items:
  sc.<m>        plurality of draws 1..5 vs draw 1                      (DeepSeek, GLM)
  sc.opus3      plurality of Opus 4.6 draws 1..3 vs draw 1 (added after the preregistration, deviation of 29.10.2026;
                in the Holm family of 9)
  copies        DeepSeek copies, plurality of round 2 vs SC@5; items without round 2 (unanimous at round 0) keep round 0
  debate7       7-model debate, plurality of round 2 vs plurality of round 0; items without round 2 keep round 0
  synth         Opus 4.6 over the 7 round-0 answers vs plurality of those answers
  decomp.<m>    decomposition chain vs draw 1 of the same model            (decomp120 items)
Participants: SC and copies — the 5 draws; debate7 and synth — the 7 round-0 answers. Definitions of C+, prize, s(A),
best(C(A)), resample headroom, CIs: docs/SETB-SPEC.md (operators/p2parts/setb.py). Costs: usd column ($ per call; Claude
notional = list price; usd_estimated = 1 for CLI-cut calls priced as a floor).
"""
import csv, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "operators"))
from p2parts import setb as SB  # noqa: E402
from p1stats.tests_stats import mcnemar_exact, holm, benjamini_hochberg  # noqa: E402

D = Path(os.environ.get("SETB_DATA", ROOT / "datasets/set_b"))   # SETB_DATA: a reproduced table
POOL = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "deepseek/deepseek-v3.2", "z-ai/glm-4.7-flash",
        "qwen/qwen3.8-27b", "moonshotai/kimi-k2.5"]
DS, GLM, OPUS = "deepseek/deepseek-v3.2", "z-ai/glm-4.7-flash", "claude-opus-4-6"
SENS = os.environ.get("SETB_NO_DEFECTIVE") == "1"
ITEMS = list(csv.DictReader(open(D / "items.csv")))
AUDIT = [{"id": r["id"], "decision": r["audit"]} for r in ITEMS if r["audit"]]
DEFECTIVE = {r["id"] for r in AUDIT if r["decision"] == "defective"}
BASE = [b for b in ITEMS if not (SENS and b["id"] in DEFECTIVE)]
IDS = [b["id"] for b in BASE]
STRAT = {b["id"]: f"{b['tier']}|{b['area']}" for b in BASE}
DEC = [b["id"] for b in sorted((b for b in BASE if b["decomp"]), key=lambda b: int(b["decomp"]))]   # decomp = position in the decomposition list (order fixes the bootstrap)

A = {}   # (id, op, model, slot, round) -> row
for r in csv.DictReader(open(D / "answers.csv")):
    r["correct"] = int(r["correct"]); r["usd"] = float(r["usd"]); r["usd_estimated"] = int(r["usd_estimated"])
    A[(r["id"], r["op"], r["model"], int(r["slot"]), int(r["round"]))] = r
CUT_EST = sum(r["usd_estimated"] for k, r in A.items() if k[0] in STRAT)


def get(i, op, m, slot=1, rnd=0):
    return A.get((i, op, m, slot, rnd))


def ok(r): return r["correct"] if r else 0
def usd(r): return r["usd"] if r else 0.0
def code(r): return r["code"] if r else ""


def vote(rows):
    return SB.plurality_codes([code(r) for r in rows], [ok(r) for r in rows])[0]


single_acc = {m: np.array([ok(get(i, "draw", m)) for i in IDS]) for m in POOL}
single_cost = {m: np.array([usd(get(i, "draw", m)) for i in IDS]) for m in POOL}
R = {"N": len(IDS), "single": {m: {"acc": float(single_acc[m].mean()), "usd": float(single_cost[m].mean())} for m in POOL}}
rng_last = None
VEC = {}   # per-item difference vectors for a maintainer check script (not published); not written to out2


def row(name, ids, a, a0, P, CA, C0, extra=None):
    global rng_last
    idx = [IDS.index(i) for i in ids]
    out, rng_last = SB.operator_stats(name, a, a0, P, CA, C0, {m: single_acc[m][idx] for m in POOL},
                                      {m: single_cost[m][idx] for m in POOL}, [STRAT[i] for i in ids], POOL,
                                      mcnemar_exact, resample=name.startswith(("sc.", "copies.")))
    if extra:
        out.update(extra() if callable(extra) else extra)
    # Bonferroni-level CIs for the prereg primary family (6 hypotheses: alpha = 0.05 / 6), own rng so the CIs above are unchanged
    a_, a0_ = np.asarray(a, float), np.asarray(a0, float)
    S = SB.splits([STRAT[i] for i in ids])
    v, _ = SB.heldout_best_under_cost({m: single_acc[m][idx] for m in POOL}, {m: single_cost[m][idx] for m in POOL}, np.asarray(CA, float), S, POOL)
    g = np.random.default_rng(SB.operator_seed(name + ":bonf")); B = g.integers(0, len(a_), size=(10000, len(a_)))
    q = lambda d: [float(d.mean()), float(np.percentile(d[B].mean(1), 100 * 0.05 / 12)), float(np.percentile(d[B].mean(1), 100 - 100 * 0.05 / 12))]
    out["bonf6"] = {"delta": q(a_ - a0_), "gain_bestC": q(a_ - v) if v is not None else None}
    VEC[name] = {"delta": (a_ - a0_).tolist(), "gain_bestC": (a_ - v).tolist() if v is not None else None}
    if np.asarray(P).shape[1] > 1:
        hb = a_ - SB.heldout_best(np.asarray(P, float), S)
        out["bonf6"]["vs_best_heldout"] = q(hb); VEC[name]["vs_best_heldout"] = hb.tolist()
    R[name] = out


draws = {m: {i: [get(i, "draw", m, k) for k in range(1, 6)] for i in IDS} for m in (DS, GLM)}
for m in (DS, GLM):
    a, a0, P, CA, C0 = [], [], [], [], []
    for i in IDS:
        rs = draws[m][i]
        a.append(vote(rs)); a0.append(ok(rs[0])); P.append([ok(r) for r in rs])
        CA.append(sum(usd(r) for r in rs)); C0.append(usd(rs[0]))
    row(f"sc.{m.split('/')[1]}", IDS, a, a0, P, CA, C0)

a, a0, a1, P, CA, C0, n_deb = [], [], [], [], [], [], 0
for i in IDS:
    rs0 = draws[DS][i]
    r1 = [get(i, "copies", DS, k, 1) for k in range(1, 6)]
    r2 = [get(i, "copies", DS, k, 2) for k in range(1, 6)]
    deb = all(r2); n_deb += deb
    a.append(vote(r2) if deb else vote(rs0)); a0.append(vote(rs0)); a1.append(ok(rs0[0])); P.append([ok(r) for r in rs0])
    base_c = sum(usd(r) for r in rs0)
    CA.append(base_c + (sum(usd(r) for r in r1 + r2) if deb else 0)); C0.append(base_c)
row("copies.deepseek", IDS, a, a0, P, CA, C0, lambda: {"debated_items": n_deb,
    "vs_draw1": {"acc0": float(np.mean(a1)), "delta": SB.bootstrap_ci(np.array(a) - np.array(a1), rng_last)}})

a, a0, a1r, P, CA, C0, n_deb, n_unan = [], [], [], [], [], [], 0, 0
for i in IDS:
    rs0 = [get(i, "draw", m) for m in POOL]
    rr = {k: [get(i, "debate7", m, 1, k) for m in POOL] for k in (1, 2)}
    deb = all(rr[2]); n_deb += deb
    v0 = vote(rs0)
    if deb and len({code(r) for r in rs0 if code(r)}) == 1:
        n_unan += 1
    a.append(vote(rr[2]) if deb else v0); a1r.append(vote(rr[1]) if deb else v0); a0.append(v0)
    P.append([ok(r) for r in rs0])
    c0 = sum(usd(r) for r in rs0)
    CA.append(c0 + (sum(usd(r) for k in (1, 2) for r in rr[k]) if deb else 0)); C0.append(c0)
deb_vec = {"_a": list(a), "_a0": list(a0), "_C": list(CA), "_C0": list(C0)}
row("debate7", IDS, a, a0, P, CA, C0, lambda: {"debated_items": n_deb, "debated_but_unanimous_by_identity": n_unan,
    "round1": {"acc": float(np.mean(a1r)), "delta": SB.bootstrap_ci(np.array(a1r) - np.array(a0), rng_last)}})

a, a0, P, CA, C0, novel = [], [], [], [], [], 0
for i in IDS:
    rs0 = [get(i, "draw", m) for m in POOL]
    s = get(i, "synth", OPUS)
    a.append(ok(s)); a0.append(vote(rs0)); P.append([ok(r) for r in rs0])
    novel += SB.new_correct_code(code(s), ok(s), [code(r) for r in rs0])
    c0 = sum(usd(r) for r in rs0); CA.append(c0 + usd(s)); C0.append(c0)
row("synth.opus", IDS, a, a0, P, CA, C0, {"correct_new_answers": novel})
syn_a, syn_C = np.asarray(a, float), np.asarray(CA, float)

# Opus 4.6 SC@3 (added 2026-10-29, not preregistered): draws 1-3; compared pairwise with vote7 / debate7 / synth
a, a0, P, CA, C0 = [], [], [], [], []
for i in IDS:
    rs = [get(i, "draw", OPUS, k) for k in (1, 2, 3)]
    a.append(vote(rs)); a0.append(ok(rs[0])); P.append([ok(r) for r in rs])
    CA.append(sum(usd(r) for r in rs)); C0.append(usd(rs[0]))
if all(get(i, "draw", OPUS, 3) for i in IDS):
    sa = np.asarray(a, float)
    def _pair(o, C):
        o = np.asarray(o, float); W = int(((sa == 0) & (o == 1)).sum()); L = int(((sa == 1) & (o == 0)).sum())
        return {"acc": float(o.mean()), "usd": float(np.mean(C)), "wins_other": W, "wins_sc3": L,
                "mcnemar_p": float(mcnemar_exact(W, L)), "delta": SB.bootstrap_ci(o - sa, rng_last)}
    row("sc.opus3", IDS, a, a0, P, CA, C0, lambda: {
        "cut_d2_d3": sum(1 for i in IDS for k in (2, 3) if get(i, "draw", OPUS, k)["finish"] == "length"),
        "draw_acc": [float(np.mean(c)) for c in zip(*P)],
        "vs": {"vote7": _pair(deb_vec["_a0"], deb_vec["_C0"]), "debate7": _pair(deb_vec["_a"], deb_vec["_C"]),
               "synth.opus": _pair(syn_a, syn_C)}})

for m in (DS, GLM, "claude-sonnet-4-6"):
    a, a0, CA, C0 = [], [], [], []
    for i in DEC:
        c = get(i, "decomp", m)
        a.append(ok(c)); a0.append(ok(get(i, "draw", m))); CA.append(usd(c)); C0.append(usd(get(i, "draw", m)))
    row(f"decomp.{m.split('/')[-1]}", DEC, a, a0, [[x] for x in a0], CA, C0)

# multiplicity: one family = the 9 operator-vs-starting-point (8 + sc.opus3, added 29.10) McNemar tests of this table (Holm; BH alongside)
FAM = [k for k in R if k not in ("N", "single")]
for k, ph, qb in zip(FAM, holm([R[k]["mcnemar_p"] for k in FAM]), benjamini_hochberg([R[k]["mcnemar_p"] for k in FAM])):
    R[k]["mcnemar_p_holm"], R[k]["mcnemar_q_bh"] = ph, qb

out = D if "SETB_DATA" in os.environ else ROOT / "summaries/out2"   # a reproduced table gets its own report
SUF = "_nodefective" if SENS else ""
json.dump(R, open(out / f"setB{SUF}.json", "w"), indent=1)
if "SETB_DATA" not in os.environ:
    (ROOT / "summaries/out3").mkdir(exist_ok=True)
    json.dump(VEC, open(ROOT / f"summaries/out3/setB{SUF}.vectors.json", "w"))   # per-item vectors for working checks

ci = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
f3 = lambda x: "—" if x is None else f"{x:+.3f}"
L = ["# Set B — operators vs starting points (paper-1 identity, held-out best, bootstrap 95% CI)", "",
     f"Generated by `operators/setb_report.py` from `datasets/set_b/` ({len(IDS)} items"
     + (", 3 defective items excluded — sensitivity" if SENS else "") + "). Claude costs are notional (list price).", "",
     f"## Single models (draw 1, {len(IDS)} items)", "", "| model | acc | $/item |", "|---|---|---|"]
for m in sorted(POOL, key=lambda m: -R["single"][m]["acc"]):
    L.append(f"| {m} | {R['single'][m]['acc']:.3f} | {R['single'][m]['usd']:.4f} |")
L += ["", "## Operators", "", "| operator | N | acc0 → acc | Δ [95% CI] | W / L | McNemar p | Holm p | BH q | C⁺ | prize | s(A) | C0 → C ($/item) | best(C) acc | gain vs best(C) | $/extra |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in R.items():
    if k in ("N", "single"):
        continue
    bc = v.get("bestC_heldout")
    has = "prize" in v
    L.append(f"| {k} | {v['N']} | {v['acc0']:.3f} → {v['acc']:.3f} | {ci(v['delta'])} | {v['W']} / {v['L']} | {v['mcnemar_p']:.3g} | {v['mcnemar_p_holm']:.3g} | {v['mcnemar_q_bh']:.3g} | "
             f"{v.get('Cplus', '—') if has else '—'} | {('%.3f' % v['prize']) if has else '—'} | {f3(v.get('s_heldout')) if has else '—'} | "
             f"{v['C0_usd']:.4f} → {v['C_usd']:.4f} | {(bc or {}).get('acc', float('nan')):.3f} | {ci(bc['gain']) if bc else '—'} | "
             f"{('%.3f' % v['usd_per_extra_item']) if v.get('usd_per_extra_item') else '—'} |")
L += ["", f"- copies: debated {R['copies.deepseek']['debated_items']} items; vs draw 1: Δ {ci(R['copies.deepseek']['vs_draw1']['delta'])}",
      f"- debate7: debated {R['debate7']['debated_items']} items, of which {R['debate7']['debated_but_unanimous_by_identity']} unanimous "
      f"under paper-1 identity; round 1: Δ {ci(R['debate7']['round1']['delta'])}",
      f"- synth: correct answers outside every input group (paper-1 identity): {R['synth.opus']['correct_new_answers']}",
      "", "## Resampling operators: share of the pass@k headroom", "", "| operator | mean draw | pass@5 | headroom | acc | share |",
      "|---|---|---|---|---|---|"]
for k, v in R.items():
    if isinstance(v, dict) and v.get("resample"):
        q = v["resample"]
        L.append(f"| {k} | {q['mean_draw']:.3f} | {q['pass_at_k']:.3f} | {q['headroom']:.3f} | {v['acc']:.3f} | "
                 f"{'—' if q['share'] is None else '%.2f' % q['share']} |")
L += [f"- Claude calls priced by estimate (CLI cut at 64k): {CUT_EST}",
      "- prize / s(A) for sc.* and copies use the best draw position as \"best\" — not meaningful; see the headroom share"]
open(out / f"setB{SUF}.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
