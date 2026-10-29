"""Operator: decomposition (v8 §9). Plan -> subtasks solved one call at a time -> assembly, at the 8,192-token budget of one
whole corpus attempt. arm2 = "split" (base model does every call), arm3 = "team" (subtask i goes to worker i mod 3 of Qwen 3.7
plus / Mistral Large / Kimi K2.5). The single-model side is the corpus: 16 draws per item at T = 1 (audited relabels; a
draw cut by the token cap is wrong).

Ported from a private script (not published) (-> STATS.md, v8 Table 9.1) with its conventions:
  * strata: selection.json stratum; "hard" split into hard-cap (>= 8 of 16 draws truncated) / hard-true.
  * single p@1 = mean over items of the mean of the 16 draws; single p@4 = any of the FIRST 4 draws; p@16 = any of 16.
  * arm p@k = any of reps 0..k-1 (hard items have reps 0-3, medium/easy rep 0 only).
  * bootstrap: paired over items, B = 2000, random.Random(0), percentile indices int(.025B), int(.975B) (stats.boot_diff).
New documented functions (keys "untraced.*") recompute the numbers of v8 §9.3-9.5 / Table 9.2 that existed only in
decomposition/RESULTS.md and the review of 2026-10-10. Decisions taken there (not in stats.py) are listed per function.
Baseline data (decided 2026-10-18): the single-model side ALWAYS reads the 8,192-capped T=1 draws
(datasets/set_a/decomposition/single-draws.csv), also under P2_DATA=uncapped: 8,192 is the equal-budget protocol of §9, so a
draw cut there is the protocol working. Override only on purpose: DRAWS_SOURCE = "uncapped" (env P2_DECOMP_DRAWS).
Budget (2026-10-19): BUDGET = env P2_DECOMP_BUDGET, 8192 | 65536. 8192 = the preregistered run (runs.jsonl, capped draws;
the paper's original numbers). 65536 = the same protocol with every budget constant x8 (total 65,536; plan 8,192; assembly
16,384; subtask floor 4,096; plan reasoning 512): runs.b65536.jsonl, and the baseline becomes the uncapped T=1 draws
(DRAWS_SOURCE forced to "uncapped"); data not published (runs.b65536.jsonl is not in this repository). Default: 8192.
No API calls; reads datasets/ only via p2parts.decomp.
"""
import collections
import math
import random
import statistics as st
from math import comb

import os

from p2parts import decomp as IO

BUDGET = int(os.environ.get("P2_DECOMP_BUDGET", "8192"))
assert BUDGET in (8192, 65536), BUDGET
DRAWS_SOURCE = "uncapped" if BUDGET == 65536 else os.environ.get("P2_DECOMP_DRAWS", "capped")
assert DRAWS_SOURCE in ("capped", "uncapped"), DRAWS_SOURCE

GLM, DS = IO.BASES
SHORT = {GLM: "glm", DS: "deepseek"}   # output keys (model ids contain dots, which break dotted claim refs)
NUMERIC = ("olympiad", "gsm-hard")


# ----------------------------------------------------------------------------------------------- data (stats.py conventions)
def load():
    sel = IO.selection()
    meta = {(b, i): (l, s, c, t) for b, l, i, s, c, t in sel}
    order = [(b, i) for b, l, i, s, c, t in sel]
    per = collections.defaultdict(list)
    drows = collections.defaultdict(list)
    for r in IO.draws(source=DRAWS_SOURCE):
        if (r["base"], r["id"]) in meta:
            per[(r["base"], r["id"])].append(r["correct"])
            drows[(r["base"], r["id"])].append(r)
    single = {k: v[:16] for k, v in per.items()}
    drows = {k: v[:16] for k, v in drows.items()}
    runs = IO.runs(budget=BUDGET)
    D = collections.defaultdict(dict)
    for r in runs:
        D[(r["base"], r["arm"], r["id"])][r["rep"]] = int(r["correct"])
    return dict(meta=meta, order=order, single=single, drows=drows, runs=runs, D=D)


def substratum(meta, b, i):
    l, s, c, t = meta[(b, i)]
    if s != "hard":
        return s
    return "hard-cap" if t >= 8 else "hard-true"


def boot_diff(xs, ys, B=2000, seed=0):
    rnd = random.Random(seed); n = len(xs); v = []
    for _ in range(B):
        idx = [rnd.randrange(n) for _ in range(n)]
        v.append(sum(xs[i] for i in idx) / n - sum(ys[i] for i in idx) / n)
    v.sort()
    return v[int(.025 * B)], v[int(.975 * B) - 1]   # same percentile convention as lib/p1stats.metrics.pct_2000


def phi(a, b_):
    n = len(a)
    if n == 0:
        return float("nan")
    p1 = sum(a) / n; p2 = sum(b_) / n
    if p1 in (0, 1) or p2 in (0, 1):
        return float("nan")
    return (sum(x and y for x, y in zip(a, b_)) / n - p1 * p2) / math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))


def arm_any(X, b, a, i, k=99):
    reps = X["D"].get((b, a, i), {})
    got = [reps[r] for r in sorted(reps) if r < k]
    return int(any(got)) if got else 0


def any_decomp(X, b, i):
    return int(any(list(X["D"].get((b, "arm2", i), {}).values()) + list(X["D"].get((b, "arm3", i), {}).values())))


def ids_of(X, b, strata=None, layers=None):
    return [i for (bb, i) in X["meta"] if bb == b and (b, i) in X["single"]
            and (strata is None or substratum(X["meta"], b, i) in strata)
            and (layers is None or X["meta"][(b, i)][0] in layers)]


def unbiased_pass(n, c, k):
    """unbiased pass@k from n draws with c correct (Chen et al. 2021): 1 - C(n-c, k)/C(n, k)."""
    if k >= n:
        return float(c > 0)
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------------------------------------- stats.py port
def stats_table(X):
    """Table of STATS.md (v8 Table 9.1): one row per (base, stratum)."""
    rows = {}
    for b in IO.BASES:
        for s in ("hard-true", "hard-cap", "medium", "easy"):
            ids = ids_of(X, b, {s})
            if not ids:
                continue
            sv = [X["single"][(b, i)] for i in ids]
            p1 = st.mean(st.mean(v) for v in sv); p4 = st.mean(any(v[:4]) for v in sv); p16 = st.mean(any(v) for v in sv)
            arm = lambda a, k: st.mean(arm_any(X, b, a, i, k) for i in ids)
            zero = [i for i in ids if sum(X["single"][(b, i)]) == 0]
            zsolved = [i for i in zero if any_decomp(X, b, i)]
            union = st.mean(int(any(X["single"][(b, i)]) or any_decomp(X, b, i)) for i in ids)
            xs = [X["D"].get((b, "arm2", i), {}).get(0, 0) for i in ids]
            ys_m = [st.mean(X["single"][(b, i)]) for i in ids]
            lo, hi = boot_diff(xs, ys_m)
            rows.setdefault(SHORT[b], {})[s] = dict(
                base=b, stratum=s, n=len(ids), single_p1=p1, single_p4=p4, single_p16=p16,
                arm2_p1=arm("arm2", 1), arm3_p1=arm("arm3", 1), arm2_p4=arm("arm2", 4), arm3_p4=arm("arm3", 4),
                both_p4=st.mean(any_decomp(X, b, i) for i in ids), zero16=len(zero), zero16_solved=len(zsolved),
                zero16_solved_ids=zsolved, union_minus_p16=union - p16, delta_p1=st.mean(xs) - p1, delta_p1_ci=[lo, hi])
    return rows


def phis(X):
    out = {}
    for b in IO.BASES:
        ids = [i for (bb, i) in X["meta"] if bb == b and (b, i) in X["single"] and 0 in X["D"].get((b, "arm2", i), {})
               and 0 in X["D"].get((b, "arm3", i), {})]
        a2 = [X["D"][(b, "arm2", i)][0] for i in ids]; a3 = [X["D"][(b, "arm3", i)][0] for i in ids]
        s0 = [X["single"][(b, i)][0] for i in ids]; s1 = [X["single"][(b, i)][1] for i in ids]
        out[SHORT[b]] = dict(base=b, n=len(ids), arm2_single0=phi(a2, s0), arm3_single0=phi(a3, s0), arm2_arm3=phi(a2, a3), single0_single1=phi(s0, s1))
    return out


def arm_counts(X):
    out = {}
    for b in IO.BASES:
        for a in ("arm2", "arm3"):
            rr = [r for r in X["runs"] if r["base"] == b and r["arm"] == a]
            out.setdefault(SHORT[b], {})[a] = dict(
                base=b, runs=len(rr), plan_fail=sum(1 for r in rr if r.get("fail")),
                asm_truncated=sum(1 for r in rr if r["calls"][-1]["step"] == "asm" and r["calls"][-1]["finish"] == "length"),
                any_sub_truncated=sum(1 for r in rr if any(c["finish"] == "length" and c["step"].startswith("sub") for c in r["calls"])),
                mean_tokens=st.mean(r["tokens"] or 0 for r in rr), mean_subtasks=st.mean(r["n_subtasks"] for r in rr))
    return out


def header(X):
    R = X["runs"]
    return dict(runs=len(R), plan_failures=sum(1 for r in R if r.get("fail")), cost=sum(r["cost"] for r in R),
                mean_tokens=st.mean(r["tokens"] or 0 for r in R))


# --------------------------------------------------------------------------------------------- untraced (RESULTS.md) numbers
def protocol_counts(X):
    """§9.3 counts: items per base/stratum/layer, DeepSeek max truncated draws on hard items, GLM hard-cap of hard,
    runs, pilot size and cost. Decision: "hard-true / hard-cap" exactly as stats.substratum (trunc16 >= 8)."""
    sel = IO.selection()
    c = collections.Counter((b, substratum(X["meta"], b, i)) for b, l, i, s, cc, t in sel)
    lay = collections.Counter((b, l) for b, l, i, s, cc, t in sel if s == "hard")
    P = IO.pilot()
    pc = sum(r["cost"] for r in P); mc = header(X)["cost"]
    return dict(
        items=len(sel), hard_items=sum(1 for r in sel if r[3] == "hard"),
        by_stratum={f"{SHORT[b]}_{s}": n for (b, s), n in sorted(c.items())},
        hard_by_layer={f"{SHORT[b]}_{l}": n for (b, l), n in sorted(lay.items())},
        ds_hard_numeric=sum(n for (b, l), n in lay.items() if b == DS and l in NUMERIC),
        ds_hard_letter=lay[(DS, "gpqa-diamond")],
        ds_hard_max_trunc16=max(t for b, l, i, s, cc, t in sel if b == DS and s == "hard"),
        ds_hard_items_with_one_trunc=sum(1 for b, l, i, s, cc, t in sel if b == DS and s == "hard" and t == 1),
        glm_hard=sum(1 for b, l, i, s, cc, t in sel if b == GLM and s == "hard"),
        glm_hard_cap=c[(GLM, "hard-cap")], glm_hard_true=c[(GLM, "hard-true")],
        runs=len(X["runs"]), runs_by_base_arm={f"{SHORT[b]}_{a}": n for (b, a), n in sorted(collections.Counter((r["base"], r["arm"]) for r in X["runs"]).items())},
        reps_hard=sorted({r["rep"] for r in X["runs"] if X["meta"][(r["base"], r["id"])][1] == "hard"}),
        reps_other=sorted({r["rep"] for r in X["runs"] if X["meta"][(r["base"], r["id"])][1] != "hard"}),
        main_cost=mc, pilot_runs=len(P), pilot_cost=pc, total_cost=mc + pc,
        max_attempt_tokens=max(r["tokens"] or 0 for r in X["runs"]),
        attempts_over_8192=sum(1 for r in X["runs"] if (r["tokens"] or 0) > 8192))


def ds_hard_true(X):
    """§9.4 "Where the model does not solve the whole task" on DeepSeek hard-true. Numeric = olympiad + gsm-hard.
    Decisions: Δ = arm2 p@4 − single p@4 (first 4 draws, the Table 9.1 convention), paired bootstrap (boot_diff, seed 0);
    union gain = mean[any(16 draws) or any of the 8 decomposition attempts (arm2 ∪ arm3)] − single p@16, its CI by the same
    bootstrap on the per-item indicators; per-attempt rate on GPQA 0/16 items = correct / attempts over arm2 + arm3 reps."""
    S = X["single"]; b = DS
    num = ids_of(X, b, {"hard-true"}, NUMERIC); gp = ids_of(X, b, {"hard-true"}, ("gpqa-diamond",))
    x4 = [arm_any(X, b, "arm2", i, 4) for i in num]; y4 = [int(any(S[(b, i)][:4])) for i in num]
    u = [int(any(S[(b, i)]) or any_decomp(X, b, i)) for i in num]; y16 = [int(any(S[(b, i)])) for i in num]
    znum = [i for i in num if sum(S[(b, i)]) == 0]; zgp = [i for i in gp if sum(S[(b, i)]) == 0]
    att = [v for i in zgp for a in ("arm2", "arm3") for v in X["D"].get((b, a, i), {}).values()]
    return dict(
        n_numeric=len(num), single_p1_numeric=st.mean(st.mean(S[(b, i)]) for i in num), single_p4_numeric=st.mean(y4),
        single_p16_numeric=st.mean(y16), arm2_p1_numeric=st.mean(arm_any(X, b, "arm2", i, 1) for i in num),
        arm2_p4_numeric=st.mean(x4), arm3_p4_numeric=st.mean(arm_any(X, b, "arm3", i, 4) for i in num),
        delta_p4_numeric=st.mean(x4) - st.mean(y4), delta_p4_numeric_ci=list(boot_diff(x4, y4)),
        union_gain_numeric=st.mean(u) - st.mean(y16), union_gain_numeric_ci=list(boot_diff(u, y16)),
        zero16_numeric=len(znum), zero16_numeric_solved=sum(any_decomp(X, b, i) for i in znum),
        zero16_numeric_attempts_each=sorted({len(X["D"].get((b, "arm2", i), {})) + len(X["D"].get((b, "arm3", i), {})) for i in znum}),
        n_gpqa=len(gp), zero16_gpqa=len(zgp), zero16_gpqa_solved=sum(any_decomp(X, b, i) for i in zgp),
        zero16_gpqa_attempts=len(att), zero16_gpqa_correct=sum(att), zero16_gpqa_rate=sum(att) / len(att),
        zero16_total=len(znum) + len(zgp), zero16_total_solved=sum(any_decomp(X, b, i) for i in znum + zgp))


def glm_cap(X):
    """§9.4 "Where a single attempt is cut by the limit" on GLM hard-cap (38).
    Decisions: Δ p@4 as in ds_hard_true; unbiased single pass@4 = mean over items of 1 − C(16−c,4)/C(16,4); per-attempt rate
    on GPQA 0/16 items over arm2 + arm3 reps; numeric union gain as in ds_hard_true on the numeric cap items."""
    S = X["single"]; b = GLM
    ids = ids_of(X, b, {"hard-cap"})
    x4 = [arm_any(X, b, "arm2", i, 4) for i in ids]; y4 = [int(any(S[(b, i)][:4])) for i in ids]
    zgp = [i for i in ids if X["meta"][(b, i)][0] == "gpqa-diamond" and sum(S[(b, i)]) == 0]
    att = [v for i in zgp for a in ("arm2", "arm3") for v in X["D"].get((b, a, i), {}).values()]
    num = [i for i in ids if X["meta"][(b, i)][0] in NUMERIC]
    u = [int(any(S[(b, i)]) or any_decomp(X, b, i)) for i in num]; y16 = [int(any(S[(b, i)])) for i in num]
    return dict(
        n=len(ids), arm2_p4=st.mean(x4), single_p4=st.mean(y4), delta_p4=st.mean(x4) - st.mean(y4),
        delta_p4_ci=list(boot_diff(x4, y4)), single_p4_unbiased=st.mean(unbiased_pass(16, sum(S[(b, i)]), 4) for i in ids),
        arm2_p1=st.mean(arm_any(X, b, "arm2", i, 1) for i in ids),
        zero16_gpqa=len(zgp), zero16_gpqa_attempts=len(att), zero16_gpqa_correct=sum(att), zero16_gpqa_rate=sum(att) / len(att),
        n_numeric=len(num), union_gain_numeric=st.mean(u) - st.mean(y16), union_gain_numeric_ci=list(boot_diff(u, y16)))


def medium_easy(X):
    """§9.4 "Where the model already coped": medium + easy pooled per base (18 items). Δ = arm2 rep-0 p@1 − mean single p@1,
    paired bootstrap on (rep-0 correct, item mean of 16 draws) exactly as the stats.py Δ column, over the pooled items in
    selection order. Truncation shares: arm2 subtask calls (step sub*) with finish = length / all arm2 subtask calls
    (review definition). "34 of 98": DeepSeek arm2 runs whose last call is the assembly, cut by length, with no parsed answer.
    Plan failures with both plan calls cut: arm2 runs with fail and every plan call finish = length."""
    out = {}
    for b in IO.BASES:
        ids = ids_of(X, b, {"medium", "easy"})
        xs = [X["D"].get((b, "arm2", i), {}).get(0, 0) for i in ids]; ys = [st.mean(X["single"][(b, i)]) for i in ids]
        out[SHORT[b]] = dict(n=len(ids), single_p1=st.mean(ys), arm2_p1=st.mean(xs), delta=st.mean(xs) - st.mean(ys),
                             loss=st.mean(ys) - st.mean(xs),
                      delta_ci=list(boot_diff(xs, ys)))
    for b in IO.BASES:
        rr = [r for r in X["runs"] if r["base"] == b and r["arm"] == "arm2"]
        sub = [c for r in rr for c in r["calls"] if c["step"].startswith("sub")]
        rr_all = [r for r in X["runs"] if r["base"] == b]
        sub_all = [c for r in rr_all for c in r["calls"] if c["step"].startswith("sub")]
        out[SHORT[b]].update(
            arm2_runs=len(rr), arm2_sub_calls=len(sub), arm2_sub_truncated=sum(c["finish"] == "length" for c in sub),
            arm2_sub_trunc_share=sum(c["finish"] == "length" for c in sub) / len(sub),
            arm2_sub_trunc_pct=100 * sum(c["finish"] == "length" for c in sub) / len(sub),
            both_arms_sub_trunc_share=sum(c["finish"] == "length" for c in sub_all) / len(sub_all),
            arm2_asm_truncated=sum(1 for r in rr if r["calls"][-1]["step"] == "asm" and r["calls"][-1]["finish"] == "length"),
            arm2_asm_truncated_no_answer=sum(1 for r in rr if r["calls"][-1]["step"] == "asm" and r["calls"][-1]["finish"] == "length"
                                             and not r["answered"]),
            arm2_plan_fail=sum(1 for r in rr if r["fail"]),
            arm2_plan_fail_all_plan_calls_length=sum(1 for r in rr if r["fail"] and all(c["finish"] == "length" for c in r["calls"] if c["step"].startswith("plan"))
                                                     and any(c["step"].startswith("plan") for c in r["calls"])))
    return out


def team_vs_split(X):
    """§9.4 "Team versus split": arm3 p@4 − arm2 p@4, paired bootstrap (boot_diff, seed 0). Cells: DeepSeek hard (20);
    GLM hard-cap numeric (olympiad + gsm-hard); GLM GPQA — decision: hard-cap GPQA items (the cap stratum, where the text
    discusses GPQA); the all-hard GPQA cell (cap + true) is reported alongside."""
    def cell(b, ids):
        x = [arm_any(X, b, "arm3", i, 4) for i in ids]; y = [arm_any(X, b, "arm2", i, 4) for i in ids]
        return dict(n=len(ids), arm3_p4=st.mean(x), arm2_p4=st.mean(y), diff=st.mean(x) - st.mean(y), ci=list(boot_diff(x, y)))
    return dict(
        ds_hard=cell(DS, ids_of(X, DS, {"hard-true", "hard-cap"})),
        glm_cap_numeric=cell(GLM, ids_of(X, GLM, {"hard-cap"}, NUMERIC)),
        glm_cap_gpqa=cell(GLM, ids_of(X, GLM, {"hard-cap"}, ("gpqa-diamond",))),
        glm_hard_gpqa=cell(GLM, ids_of(X, GLM, {"hard-cap", "hard-true"}, ("gpqa-diamond",))))


def dollars(X):
    """Table 9.2 (RESULTS.md "Dollars", as corrected by the review 2026-10-10). Per (base, stratum):
    draw $ / tokens = mean cost / mean out tokens over the 16 corpus draws of every item; arm $ / tokens = mean over all
    attempts of that arm in the stratum; ratio = arm mean $ / draw mean $. Equal money: k1 = round(arm2 mean $ / draw $),
    k4 = round(mean over items of the summed $ of reps 0-3 / draw $) (hard strata); single pass@k = unbiased from the 16 draws.
    Python round() (banker's) on the ratio, floored at k = 1 (a single model cannot spend less than one draw;
    GLM medium has ratio 0.42 and RESULTS.md prints k = 1). Hard-true numeric row: DeepSeek hard-true olympiad + gsm-hard."""
    out = {}
    cells = [(DS, "hard-true", None), (DS, "hard-true-numeric", NUMERIC), (DS, "medium", None), (DS, "easy", None),
             (GLM, "hard-cap", None), (GLM, "medium", None), (GLM, "easy", None)]
    for b, s, lay in cells:
        ids = ids_of(X, b, {s.replace("-numeric", "")}, lay)
        dr = [d for i in ids for d in X["drows"][(b, i)]]
        dcost = st.mean(d["cost"] for d in dr); dtok = st.mean(d["out_tokens"] for d in dr)
        row = dict(n=len(ids), draw_cost=dcost, draw_tokens=dtok)
        for a in ("arm2", "arm3"):
            rr = [r for r in X["runs"] if r["base"] == b and r["arm"] == a and r["id"] in ids]
            row[f"{a}_cost"] = st.mean(r["cost"] for r in rr); row[f"{a}_tokens"] = st.mean(r["tokens"] or 0 for r in rr)
            row[f"{a}_ratio"] = row[f"{a}_cost"] / dcost
        sums = [sum(r["cost"] for r in X["runs"] if r["base"] == b and r["arm"] == "arm2" and r["id"] == i) for i in ids]
        row["k1_raw"] = row["arm2_ratio"]; row["k1"] = max(1, round(row["k1_raw"]))
        row["single_at_k1"] = st.mean(unbiased_pass(16, sum(X["single"][(b, i)]), row["k1"]) for i in ids)
        row["arm2_p1"] = st.mean(arm_any(X, b, "arm2", i, 1) for i in ids)
        if s.startswith("hard"):
            row["k4_raw"] = st.mean(sums) / dcost; row["k4"] = max(1, round(row["k4_raw"]))
            row["single_at_k4"] = st.mean(unbiased_pass(16, sum(X["single"][(b, i)]), row["k4"]) for i in ids)
            row["arm2_p4"] = st.mean(arm_any(X, b, "arm2", i, 4) for i in ids)
        out.setdefault(SHORT[b], {})[s] = row
    return out


def grader_artifact(X, item="gsmhard:11771746"):
    """§9.5: the GLM hard-true item whose single answers are all one value just outside the numeric tolerance.
    Reports the 16 single answers, the gold and the arm2/arm3 answers with correctness; and the GLM hard-true items where
    split p@4 is 1 while single p@4 (first 4 draws) is 0 (the "two wins")."""
    b = GLM
    sa = [d["answer"] for d in X["drows"][(b, item)]]
    rr = [r for r in X["runs"] if r["base"] == b and r["id"] == item]
    wins = [i for i in ids_of(X, b, {"hard-true"}) if arm_any(X, b, "arm2", i, 4) and not any(X["single"][(b, i)][:4])]
    return dict(single_answers=collections.Counter(sa).most_common(), single_correct=sum(X["single"][(b, item)]),
                gold=rr[0]["gold"], decomp_correct_answers=sorted({r["answer"] for r in rr if r["correct"]}),
                decomp_wrong_answers=sorted({r["answer"] for r in rr if not r["correct"]}),
                glm_hard_true_wins=wins, n_single_mode=collections.Counter(sa).most_common(1)[0][1],
                glm_hard_true_any_decomp_not_single_p4=[i for i in ids_of(X, b, {"hard-true"}) if any_decomp(X, b, i) and not any(X["single"][(b, i)][:4])],
                n_glm_hard_true_any_decomp_not_single_p4=sum(1 for i in ids_of(X, b, {"hard-true"}) if any_decomp(X, b, i) and not any(X["single"][(b, i)][:4])),
                glm_hard_true_any_decomp_not_single16=[i for i in ids_of(X, b, {"hard-true"}) if any_decomp(X, b, i) and not any(X["single"][(b, i)])])


def run() -> dict:
    X = load()
    res = dict(header=header(X), table=stats_table(X), phi=phis(X), arms=arm_counts(X), untraced=dict(
        protocol=protocol_counts(X), ds_hard_true=ds_hard_true(X), glm_cap=glm_cap(X), medium_easy=medium_easy(X),
        team_vs_split=team_vs_split(X), dollars=dollars(X), grader_artifact=grader_artifact(X)))
    return {"decomposition": res, "decomposition_stats.md": stats_md(res)}


def stats_md(res):
    """STATS.md as stats.py prints it (regression target)."""
    h = res["header"]
    out = ["# stats (generated)", "", f"runs {h['runs']}; plan failures {h['plan_failures']}; cost ${h['cost']:.2f}; mean tokens {h['mean_tokens']:.0f}", ""]
    out.append("| base | stratum | n | single p@1 | single p@4 | single p@16 | arm2 p@1 | arm3 p@1 | arm2 p@4 | arm3 p@4 | arm2∪arm3 @4 | 0/16 items | 0/16 solved by any decomp | union(single16 ∪ decomp) − single16 | Δ arm2 p@1 − single p@1 [CI] |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in (r for t in res["table"].values() for r in t.values()):
        z = r["zero16_solved_ids"]; lo, hi = r["delta_p1_ci"]
        out.append(f"| {r['base'].split('_')[1][:8]} | {r['stratum']} | {r['n']} | {r['single_p1']:.2f} | {r['single_p4']:.2f} | {r['single_p16']:.2f} | {r['arm2_p1']:.2f} | {r['arm3_p1']:.2f} | {r['arm2_p4']:.2f} | {r['arm3_p4']:.2f} | {r['both_p4']:.2f} | {r['zero16']} | {len(z)} {z if z else ''} | {r['union_minus_p16']:+.2f} | {r['delta_p1']:+.2f} [{lo:+.2f},{hi:+.2f}] |")
    out.append("")
    for p in res["phi"].values():
        out.append(f"{p['base']}: n={p['n']} φ(arm2,single0)={p['arm2_single0']:.2f} φ(arm3,single0)={p['arm3_single0']:.2f} φ(arm2,arm3)={p['arm2_arm3']:.2f} φ(single0,single1)={p['single0_single1']:.2f}")
    out.append("")
    for arm, a in ((arm, a) for t in res["arms"].values() for arm, a in t.items()):
        b = a["base"]
        out.append(f"{b} {arm}: runs {a['runs']}, plan fail {a['plan_fail']}, asm truncated {a['asm_truncated']}, any subtask truncated {a['any_sub_truncated']}, mean tokens {a['mean_tokens']:.0f}, mean subtasks {a['mean_subtasks']:.1f}")
    return "\n".join(out)


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r["decomposition"]["untraced"], indent=1, default=str)[:20000])
