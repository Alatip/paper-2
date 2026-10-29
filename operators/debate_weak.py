"""Debate, synthesis and judges on the non-frontier pool W (v8 §6.4, Table 6.2; restated in §6.7 and §7.5).

W = DeepSeek V3.2, GLM-5.3-flash, Mistral Large 2512, Claude Haiku 4.5 (the paper's debate order restricted to W).

Method (ported from the weak-pool analysis scripts of tech-note 5, not published; their helpers are copied below):
  * Item sets (datasets/set_a/debx-weak-pool/items.jsonl): the 733 items answered by all seven corpus models (TN5 raw_matrix, label
    audit applied); contested-for-W (>= 1 W right and >= 1 wrong at round 0), all-W-wrong, and a seeded control of 40 all-W-right
    items (random.Random(0).sample of the sorted all-right ids).
  * Debate (debate-w): round 0 = the stored matrix answers; rounds 1-3 new synchronous rounds. A row with correct = None is a
    cut-off (no FINAL line) and is a missing vote. Plurality on canon(answer) under three tie rules: first in the fixed order W,
    strongest first (W sorted by round-0 accuracy on the completed contested items, in-sample), ties split (1/k credit).
    Fixed / broken = plurality 0 -> 1 / 1 -> 0 between round 0 and round t; exact two-sided McNemar on (fixed, broken).
    Mean pairwise φ over W on the 0/1 correctness (cut-off = missing, pairwise-complete, pairs with >= 5 items and both
    marginals non-degenerate); n_eff = 4 / (1 + 3 φ̄).
  * Projection to the 733: the 577 all-W-right items outside the control are scored right (control broke 0 of 40), the 10
    all-W-wrong and the 146 contested items take their round-0 / round-3 strongest-first plurality.
  * Flips: a correct agent at round t answering at t + 1, by the number of answering peers holding a different canon answer.
  * Synthesis (synth-w): one DeepSeek call reads the four round-0 answers; graded as stored.
  * J3 panel (judge-w): Haiku, DeepSeek, GLM pick among the W candidates; unnamed pick = wrong; panel plurality of the named
    picks under the same three tie rules (strongest judge = highest pick accuracy, in-sample).
  * Contaminated control (§6.4 "What cannot be measured"): the same tables for the W votes inside the stored six-agent debate
    (datasets/set_a/debx-six-debate, TN5 load_debate, 225 items complete in all rounds), where W saw Sonnet and Qwen in rounds 1-3.
  * Six-agent flips (the "29 percent" of §6.4): TN5 stats.section_debate_deep flips on the same 225 items.

Implementation decisions:
  * Answer identity (decision 2026-10-18): default = paper 1's rule (p2parts/answer_id.py: 12-digit canon + decimal rounding,
    chained groups per vote; a group is correct if any member is). canon() below is TN5 stats.canon (7 significant digits),
    used only with P2_ANSWER_RULE=v8 (reproduces v8). Tie rules unchanged (first / strongest-first / split).
  * GPQA answers: TN5 stored option hashes; the dataset holds the matching letters (one-to-one per item), so every plurality /
    unanimity count is unchanged. Only one diagnostic of the original output changes: "synthesis picked an answer equal to
    DeepSeek's own" compared a letter with a hash on GPQA in the original (never equal) and now compares letters
    (key synth.contested.equal_own; not a v8 number).
  * Answers that are reasoning fragments were replaced by an identity-preserving code (hash of canon(answer)) in the dataset.
"""
import collections
import itertools
import math
import random

from p2parts import debx as IO
from p2parts import answer_id as AID

W = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "claude:haiku"]
W3 = W[:3]
A6 = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "qwen/qwen3.7-plus", "claude:haiku", "claude:sonnet55"]
J3 = ["claude:haiku", "deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash"]
HARD = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]
SH = {"deepseek/deepseek-v3.2": "DeepSeek", "z-ai/glm-5.3-flash": "GLM", "mistralai/mistral-large-2512": "Mistral",
      "qwen/qwen3.7-plus": "Qwen", "claude:haiku": "Haiku", "claude:sonnet55": "Sonnet"}
N_ITEMS = 733


# ----------------------------------------------------------------------------- TN5 stats.py helpers (verbatim logic)
def canon(a):
    import re
    if a in (None, ""): return a
    t = str(a).strip().replace("$", "").replace(" ", "")
    t = re.sub(r"^\\\(|\\\)$", "", t); t = re.sub(r"(?<=\d),(?=\d{3})", "", t).rstrip(".")
    try: return repr(float(f"{float(t):.7g}"))
    except ValueError: pass
    t = t.replace("\\dfrac", "\\frac"); t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
    t = t.replace("\\sqrt{", "√").replace("\\sqrt", "√").replace("{", "").replace("}", "").lower()
    return t


def present(r): return bool(r) and r.get("correct") is not None
def cval(r): return bool(r.get("correct")) if present(r) else None


def phi(a, b):
    n = len(a); p1 = sum(a) / n; p2 = sum(b) / n
    if p1 in (0, 1) or p2 in (0, 1): return None
    p12 = sum(x and y for x, y in zip(a, b)) / n
    return (p12 - p1 * p2) / math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))


def tetra(a, b):
    n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum((not x) and y for x, y in zip(a, b))
    if n10 * n01 == 0 or n11 * n00 == 0: return None
    return math.cos(math.pi / (1 + math.sqrt(n11 * n00 / (n10 * n01))))


def pairwise_phi(cols):
    ph = []
    for a, b in itertools.combinations(list(cols), 2):
        x = [(u, v) for u, v in zip(cols[a], cols[b]) if u is not None and v is not None]
        if len(x) < 5: continue
        q = phi([u for u, _ in x], [v for _, v in x])
        if q is not None: ph.append(q)
    return (sum(ph) / len(ph) if ph else None), len(ph)


def n_eff(n, ph): return n / (1 + (n - 1) * ph) if ph is not None else None


def majority(answers):
    if AID.RULE == "v8":
        c = collections.Counter(canon(a) for a in answers if a not in (None, ""))
        if not c: return None, 0
        return c.most_common(1)[0]
    top, cnt, _, _ = AID.vote(answers)
    return (top, cnt) if top is not None else (None, 0)


def keys(answers):
    """Identity key per answer of ONE vote (v8: canon; p1: paper-1 group id)."""
    return AID.keys(answers, canon)


def maj_correct(rs):
    rs = [r for r in rs if present(r)]
    if AID.RULE == "v8":
        top, _ = majority([r.get("answer") for r in rs])
        if top is None: return False
        return bool(next((r for r in rs if canon(r.get("answer")) == top), {}).get("correct"))
    top, _, ok, _ = AID.vote([r.get("answer") for r in rs], [r.get("correct") for r in rs])
    return bool(ok) if top is not None else False


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def plur(rs, rule, order=None):
    """Plurality over answering rows; 'first' (ties to first row in order) or 'split' (1/k)."""
    rs = [r for r in (rs if order is None else [rs[a] for a in order]) if present(r)]
    if not rs: return 0.0
    k = keys([r["answer"] for r in rs])
    c = collections.Counter(k); top = max(c.values()); tied = [a for a in c if c[a] == top]
    right = {x for x, r in zip(k, rs) if r["correct"]}
    if rule == "split": return sum(a in right for a in tied) / len(tied)
    first = next(x for x in k if c[x] == top)
    return float(first in right)


def has_tie(rs):
    rs = [r for r in rs if present(r)]
    if not rs: return False
    c = collections.Counter(keys([r["answer"] for r in rs])); top = max(c.values())
    return sum(k == top for k in c.values()) > 1


# ----------------------------------------------------------------------------- data
def _data():
    items = IO.weak_items()
    chosen = {r["id"]: r for r in items if r["group"] != "allright"}
    R0 = collections.defaultdict(dict)
    for r in IO.weak_round0(): R0[r["id"]][r["agent"]] = r
    return items, chosen, R0


def _load_debate(chosen, R0):
    D = collections.defaultdict(dict)
    for i in chosen:
        for a in W: D[i][(a, 0)] = R0[i][a]
    for r in IO.weak_debate(): D[r["id"]][(r["agent"], r["round"])] = r
    full = {i: d for i, d in D.items() if all((a, rd) in d for a in W for rd in range(4))}
    return D, full


def _block(full, ids, chosen, strong):
    ids = sorted(ids); n = len(ids)
    res, rounds = {}, []
    for rd in range(4):
        rows = {i: {a: full[i][(a, rd)] for a in W} for i in ids}
        pf = {i: plur(rows[i], "first", W) for i in ids}; ps = {i: plur(rows[i], "first", strong) for i in ids}; pe = {i: plur(rows[i], "split", W) for i in ids}
        res[rd] = dict(pf=pf, ps=ps, pe=pe)
        pr = {i: [r for r in rows[i].values() if present(r)] for i in ids}
        acc_n = sum(bool(r.get("correct")) for i in ids for r in pr[i]); acc_d = sum(len(pr[i]) for i in ids)
        una = unw = allw = 0
        for i in ids:
            top, cnt = majority([r.get("answer") for r in pr[i]])
            if len(pr[i]) >= 2 and cnt == len(pr[i]):
                una += 1; unw += not maj_correct(pr[i])
            allw += bool(pr[i]) and not any(r.get("correct") for r in pr[i])
        mphi, nph = pairwise_phi({a: [cval(full[i][(a, rd)]) for i in ids] for a in W})
        fb = lambda k: (sum(1 for i in ids if res[0][k][i] == 0 and res[rd][k][i] == 1), sum(1 for i in ids if res[0][k][i] == 1 and res[rd][k][i] == 0))
        (a1, b1), (a2, b2) = fb("pf"), fb("ps")
        rounds.append(dict(agent_acc=acc_n / max(acc_d, 1), plur_first=sum(pf.values()) / n, plur_strong=sum(ps.values()) / n,
                           plur_split=sum(pe.values()) / n, ties=sum(has_tie(list(rows[i].values())) for i in ids), unanimous=una,
                           fixed_first=a1, broken_first=b1, fixed_strong=a2, broken_strong=b2, mean_phi=mphi, phi_pairs=nph,
                           n_eff=n_eff(4, mphi), all_wrong=allw, unanimous_wrong=unw,
                           cutoff_rows=sum(r.get("correct") is None for i in ids for r in rows[i].values())))
    g = lambda k: sum(res[3][k].values()) / n - sum(res[0][k].values()) / n
    fx = {k: [i for i in ids if res[0][k][i] == 0 and res[3][k][i] == 1] for k in ("pf", "ps")}
    br = {k: [i for i in ids if res[0][k][i] == 1 and res[3][k][i] == 0] for k in ("pf", "ps")}
    per = {}
    for layer in HARD:
        li = [i for i in ids if chosen[i]["layer"] == layer]
        if li: per[layer] = {"n": len(li), **{f"{k}_r{rd}": sum(res[rd][k][i] for i in li) / len(li) for k in ("pf", "ps", "pe") for rd in (0, 3)}}
    out = dict(n=n, layers={l: c for l, c in sorted(collections.Counter(chosen[i]["layer"] for i in ids).items())}, rounds=rounds,
               gain_first=g("pf"), gain_strong=g("ps"), gain_split=g("pe"),
               fixed_first=len(fx["pf"]), broken_first=len(br["pf"]), p_first=mcnemar(len(fx["pf"]), len(br["pf"])),
               fixed_strong=len(fx["ps"]), broken_strong=len(br["ps"]), p_strong=mcnemar(len(fx["ps"]), len(br["ps"])),
               fixed_first_tie_r0=sum(has_tie([full[i][(a, 0)] for a in W]) for i in fx["pf"]), per_layer=per)
    return out, res


def section_debate(chosen, R0):
    D, full = _load_debate(chosen, R0)
    con = [i for i in full if chosen[i]["group"] == "contested"]
    acc = {a: sum(bool(full[i][(a, 0)].get("correct")) for i in con) / len(con) for a in W}
    strong = sorted(W, key=lambda a: -acc[a])
    out = dict(rows_collected=sum(1 for i in D for k in D[i] if k[1] > 0), rows_planned=3 * 4 * len(chosen), complete=len(full),
               strong_order=[SH[a] for a in strong], strong_acc={SH[a]: acc[a] for a in strong}, groups={})
    results = {}
    for g in ("contested", "control", "allwrong", "all"):
        ids = [i for i in full if g == "all" or chosen[i]["group"] == g]
        b, res = _block(full, ids, chosen, strong); out["groups"][g] = b; results[g] = (ids, res)
    ids, res = results["all"]
    out["members"] = {}
    for a in W:
        m = {i: bool(full[i][(a, 0)].get("correct")) for i in ids}
        b = sum(1 for i in ids if m[i] and res[3]["pf"][i] == 0); c = sum(1 for i in ids if not m[i] and res[3]["pf"][i] == 1)
        b2 = sum(1 for i in ids if m[i] and res[3]["ps"][i] == 0); c2 = sum(1 for i in ids if not m[i] and res[3]["ps"][i] == 1)
        out["members"][SH[a]] = dict(acc_r0=sum(m.values()) / len(ids), first_lost=b, first_won=c, p_first=mcnemar(b, c),
                                     strong_lost=b2, strong_won=c2, p_strong=mcnemar(b2, c2))
    allright_n = N_ITEMS - sum(1 for i in chosen if chosen[i]["group"] != "control")
    (_, cres), (ctl_ids, tres), (_, ares) = results["contested"], results["control"], results["allwrong"]
    br_n = sum(1 for i in ctl_ids if tres[3]["ps"][i] == 0); br_rate = br_n / len(ctl_ids)
    out["projection"] = dict(allright_undebated=allright_n,
                             r0=(allright_n + sum(cres[0]["ps"].values()) + sum(ares[0]["ps"].values())) / N_ITEMS,
                             r3_nobreak=(allright_n + sum(cres[3]["ps"].values()) + sum(ares[3]["ps"].values())) / N_ITEMS,
                             r3_break=(allright_n * (1 - br_rate) + sum(cres[3]["ps"].values()) + sum(ares[3]["ps"].values())) / N_ITEMS,
                             control_broken=br_n, control_n=len(ctl_ids), control_wilson=list(wilson(br_n, len(ctl_ids))))
    fl = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        for rd in range(3):
            for a in W:
                r = full[i][(a, rd)]
                if not r.get("correct"): continue
                nxt = full[i][(a, rd + 1)]
                if not present(nxt): continue
                peers = [o for o in W if o != a and present(full[i][(o, rd)])]
                k = keys([r.get("answer")] + [full[i][(o, rd)].get("answer") for o in peers])
                opp = sum(1 for x in k[1:] if x != k[0])
                fl[opp][0] += 1; fl[opp][1] += not nxt.get("correct")
    out["flips"] = {str(k): dict(cases=v[0], flipped=v[1], share=v[1] / v[0]) for k, v in sorted(fl.items())}
    tk = collections.defaultdict(list)
    for i in ids:
        for rd in range(1, 4):
            for a in W:
                t = full[i][(a, rd)].get("tok_out")
                if t: tk[a].append(t)
    out["tokens_mean"] = {SH[a]: sum(v) / len(v) for a, v in tk.items()}
    return out, full, results, strong


def section_synth(chosen, R0, results, strong):
    rows = {r["id"]: r for r in IO.weak_synth()}
    out = {}
    for g in ("contested", "allwrong"):
        ids = sorted(i for i in rows if chosen[i]["group"] == g)
        ok = sum(bool(rows[i]["correct"]) for i in ids); lo, hi = wilson(ok, len(ids))
        o = dict(n=len(ids), correct=ok, acc=ok / len(ids), wilson=[lo, hi], unparsed=sum(rows[i]["correct"] is None for i in ids),
                 novel=sum(bool(rows[i]["novel"]) for i in ids), novel_correct=sum(bool(rows[i]["novel"]) and bool(rows[i]["correct"]) for i in ids))
        if g == "contested":
            r0 = {i: {a: R0[i][a] for a in W} for i in ids}
            for rule, order, nm in (("first", W, "first"), ("first", strong, "strong"), ("split", W, "split")):
                p0 = {i: plur(r0[i], rule, order) for i in ids}
                b = sum(1 for i in ids if p0[i] == 1 and not rows[i]["correct"]); c = sum(1 for i in ids if p0[i] == 0 and rows[i]["correct"])
                o[f"vs_plur_{nm}"] = dict(plur=sum(p0.values()) / len(ids), plur_right_synth_wrong=b, reverse=c, p=mcnemar(b, c))
            o["deepseek_own_acc"] = sum(bool(R0[i]["deepseek/deepseek-v3.2"]["correct"]) for i in ids) / len(ids)
            o["equal_own"] = sum((lambda k: k[0] is not None and k[0] == k[1])(keys([rows[i]["answer"], R0[i]["deepseek/deepseek-v3.2"]["answer"]]))
                                 if AID.RULE != "v8" else canon(rows[i]["answer"]) == canon(R0[i]["deepseek/deepseek-v3.2"]["answer"]) for i in ids)
            cres = results["contested"][1]; both = [i for i in ids if i in cres[3]["ps"]]
            o["debate_r3_strong_on_both"] = sum(cres[3]["ps"][i] for i in both) / len(both)
            o["synth_on_both"] = sum(bool(rows[i]["correct"]) for i in both) / len(both)
        out[g] = o
    out["rows"] = len(rows); out["cost"] = sum(r.get("cost") or 0 for r in rows.values())
    return out


def section_judge(chosen, R0, results, strong):
    by = collections.defaultdict(dict)
    for r in IO.weak_judge(): by[r["id"]][r["judge"]] = r
    ids = sorted(i for i in by if chosen[i]["group"] == "contested" and all(j in by[i] for j in J3))
    aw = sorted(i for i in by if chosen[i]["group"] == "allwrong" and all(j in by[i] for j in J3))
    n = len(ids)
    out = dict(n=n, n_allwrong=len(aw), cands={str(k): v for k, v in sorted(collections.Counter(by[i][J3[0]]["n_cands"] for i in ids).items())}, judges={})
    pa = {}
    for j in J3:
        v = [by[i][j] for i in ids]; named = [r for r in v if r.get("pick_answer") not in (None, "")]
        ow = [r for r in v if r.get("own_correct") is False]
        pa[j] = sum(bool(r["pick_correct"]) for r in v) / n
        out["judges"][SH[j]] = dict(pick_acc=pa[j], named=len(named), named_acc=sum(bool(r["pick_correct"]) for r in named) / max(len(named), 1),
                                    own_acc=sum(bool(r["own_correct"]) for r in v) / n, picked_own=sum(r["picked_own"] for r in v) / n,
                                    picked_own_when_wrong=sum(r["picked_own"] for r in ow) / max(len(ow), 1), own_wrong_n=len(ow),
                                    none_on_allwrong=sum(by[i][j]["pick"] == "NONE" for i in aw))
    cols = {j: [bool(by[i][j]["pick_correct"]) for i in ids] for j in J3}
    ph = [p for a, b in itertools.combinations(J3, 2) if (p := phi(cols[a], cols[b])) is not None]
    te = [t for a, b in itertools.combinations(J3, 2) if (t := tetra(cols[a], cols[b])) is not None]
    mphi = sum(ph) / len(ph) if ph else None
    out.update(mean_phi=mphi, mean_tetra=sum(te) / len(te) if te else None, n_eff=n_eff(3, mphi))
    jstrong = sorted(J3, key=lambda j: -pa[j])

    def pv(i, order):
        js = [j for j in order if by[i][j].get("pick_answer") not in (None, "")]
        return list(zip(keys([by[i][j]["pick_answer"] for j in js]), [bool(by[i][j]["pick_correct"]) for j in js]))

    def pl(votes, rule, size=3):
        if not votes: return 0.0 if rule != "strict" else None
        cnt = collections.Counter(v for v, _ in votes); top = max(cnt.values()); tied = [v for v in cnt if cnt[v] == top]; right = {v for v, ok in votes if ok}
        if rule == "split": return sum(v in right for v in tied) / len(tied)
        if rule == "strict": return None if top <= size / 2 else float(tied[0] in right)
        return float(next(v for v, _ in votes if cnt[v] == top) in right)
    r0 = {i: {a: R0[i][a] for a in W} for i in ids}
    out["tie_rules"] = {}
    for lab, so, jo, rule in (("first", W, J3, "first"), ("split", W, J3, "split"), ("strong", strong, jstrong, "first")):
        sp = {i: plur(r0[i], rule, so) for i in ids}; pp = {i: pl(pv(i, jo), rule) for i in ids}
        won = sum(1 for i in ids if pp[i] == 1 and sp[i] == 0); lost = sum(1 for i in ids if pp[i] == 0 and sp[i] == 1)
        out["tie_rules"][lab] = dict(solvers=sum(sp.values()) / n, panel=sum(pp.values()) / n, diff=sum(pp.values()) / n - sum(sp.values()) / n,
                                     won=won, lost=lost, p=mcnemar(won, lost))
    strict = {i: pl(pv(i, J3), "strict") for i in ids}; cov = [i for i in ids if strict[i] is not None]; k = sum(strict[i] for i in cov)
    sp_s = {i: plur(r0[i], "first", strong) for i in ids}
    out["strict"] = dict(answered=len(cov), acc=k / max(len(cov), 1), wilson=list(wilson(k, len(cov))),
                         plur_on_answered=sum(sp_s[i] for i in cov) / max(len(cov), 1), plur_on_abstained=sum(sp_s[i] for i in ids if i not in cov) / max(n - len(cov), 1))
    cres = results["contested"][1]; both = [i for i in ids if i in cres[3]["ps"]]
    pp = {i: pl(pv(i, jstrong), "first") for i in both}
    won = sum(1 for i in both if pp[i] == 1 and cres[3]["ps"][i] == 0); lost = sum(1 for i in both if pp[i] == 0 and cres[3]["ps"][i] == 1)
    out["vs_debate"] = dict(n=len(both), panel=sum(pp.values()) / len(both), debate=sum(cres[3]["ps"][i] for i in both) / len(both), won=won, lost=lost, p=mcnemar(won, lost))
    allright_n = N_ITEMS - sum(1 for i in chosen if chosen[i]["group"] != "control")
    base = {a: 0 for a in W}
    for i, d in R0.items():
        for a in W: base[a] += bool(d[a]["correct"])
    bestW = max(W, key=lambda a: base[a])
    p0f = sum(plur(r0[i], "first", W) for i in ids); p0s = sum(plur(r0[i], "first", strong) for i in ids)
    panf = sum(pl(pv(i, J3), "first") for i in ids); pans = sum(pl(pv(i, jstrong), "first") for i in ids)
    panfb = sum((pl(pv(i, J3), "first") if pv(i, J3) and not (lambda c: sum(x == max(c.values()) for x in c.values()) > 1)(collections.Counter(v for v, _ in pv(i, J3))) else plur(r0[i], "first", W)) for i in ids)
    out["whole_733"] = dict(plur_first=(allright_n + p0f) / N_ITEMS, plur_strong=(allright_n + p0s) / N_ITEMS, panel_first=(allright_n + panf) / N_ITEMS,
                            panel_fallback=(allright_n + panfb) / N_ITEMS, panel_strong=(allright_n + pans) / N_ITEMS,
                            best_member=SH[bestW], best_member_acc=base[bestW] / N_ITEMS, deepseek_acc=base["deepseek/deepseek-v3.2"] / N_ITEMS,
                            oracle=(N_ITEMS - sum(1 for i in chosen if chosen[i]["group"] == "allwrong")) / N_ITEMS,
                            member_acc={SH[a]: base[a] / N_ITEMS for a in W})
    out["rows"] = sum(len(d) for d in by.values()); out["cost"] = sum(r.get("cost") or 0 for d in by.values() for r in d.values())
    return out


def section_spend():
    out = {}; T = collections.defaultdict(float)
    for d, rows in (("debate-w", IO.weak_debate()), ("synth-w", IO.weak_synth()), ("judge-w", IO.weak_judge())):
        o = h = 0.0
        for r in rows:
            if str(r.get("served", "")).startswith("claude"): h += r.get("cost") or 0
            else: o += r.get("cost") or 0
        out[d] = dict(rows=len(rows), openrouter=o, haiku_cli=h); T["o"] += o; T["h"] += h
    out["total"] = dict(openrouter=T["o"], haiku_cli=T["h"], all=T["o"] + T["h"])
    return out


# ----------------------------------------------------------------------------- contaminated control: W inside the six-agent debate
def _six_full():
    full = collections.defaultdict(dict)
    for r in IO.six_debate(): full[r["id"]][(r["agent"], r["round"])] = r
    return dict(full)


def six_table(full, pool, ids=None):
    """Debate table of the weak-pool analysis: plurality via maj_correct (ties to the first answering row)."""
    ids = sorted(ids if ids is not None else full); n = len(ids)
    acc0 = {a: sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in pool}
    strong = sorted(pool, key=lambda a: -acc0[a])
    res, rounds = {}, []
    for rd in range(4):
        rows = {i: [full[i][(a, rd)] for a in pool] for i in ids}
        rows_s = {i: [full[i][(a, rd)] for a in strong] for i in ids}
        pr = {i: [r for r in rows[i] if present(r)] for i in ids}
        pf = {i: float(maj_correct(rows[i])) for i in ids}; ps = {i: float(maj_correct(rows_s[i])) for i in ids}
        pe = {i: plur(rows[i], "split") for i in ids}
        una = unw = allw = 0
        for i in ids:
            top, cnt = majority([r.get("answer") for r in pr[i]])
            if len(pr[i]) >= 2 and cnt == len(pr[i]):
                una += 1; unw += not maj_correct(pr[i])
            allw += bool(pr[i]) and not any(r.get("correct") for r in pr[i])
        mphi, nph = pairwise_phi({a: [cval(full[i][(a, rd)]) for i in ids] for a in pool})
        res[rd] = dict(pf=pf, ps=ps, pe=pe)
        rounds.append(dict(agent_acc=sum(bool(r.get("correct")) for i in ids for r in pr[i]) / max(sum(len(pr[i]) for i in ids), 1),
                           plur_first=sum(pf.values()) / n, plur_strong=sum(ps.values()) / n, plur_split=sum(pe.values()) / n,
                           ties=sum(has_tie(rows[i]) for i in ids), unanimous=una, mean_phi=mphi, phi_pairs=nph, n_eff=n_eff(len(pool), mphi),
                           all_wrong=allw, unanimous_wrong=unw, cutoff_rows=sum(r.get("correct") is None for i in ids for r in rows[i])))
    fx = [i for i in ids if res[0]["pf"][i] == 0 and res[3]["pf"][i] == 1]; br = [i for i in ids if res[0]["pf"][i] == 1 and res[3]["pf"][i] == 0]
    fxs = [i for i in ids if res[0]["ps"][i] == 0 and res[3]["ps"][i] == 1]; brs = [i for i in ids if res[0]["ps"][i] == 1 and res[3]["ps"][i] == 0]
    cut0 = {i for i in ids if any(not present(full[i][(a, 0)]) for a in pool)}
    return dict(n=n, strong_order=[SH[a] for a in strong], strong_acc={SH[a]: acc0[a] for a in strong}, rounds=rounds,
                gain_first=rounds[3]["plur_first"] - rounds[0]["plur_first"], gain_strong=rounds[3]["plur_strong"] - rounds[0]["plur_strong"],
                gain_split=rounds[3]["plur_split"] - rounds[0]["plur_split"],
                fixed_first=len(fx), broken_first=len(br), p_first=mcnemar(len(fx), len(br)),
                fixed_strong=len(fxs), broken_strong=len(brs), p_strong=mcnemar(len(fxs), len(brs)),
                cut0=len(cut0), fixed_first_on_cut0=sum(i in cut0 for i in fx), fixed_strong_on_cut0=sum(i in cut0 for i in fxs),
                fixed_first_tie_r0=sum(has_tie([full[i][(a, 0)] for a in pool]) for i in fx))


def six_flips(full):
    """TN5 stats.section_debate_deep flips, six agents, 225 items."""
    fl = collections.defaultdict(lambda: [0, 0])
    for i in full:
        for rd in range(3):
            for a in A6:
                r = full[i][(a, rd)]
                if not r.get("correct"): continue
                nxt = full[i][(a, rd + 1)]
                if not present(nxt): continue
                peers = [o for o in A6 if o != a and present(full[i][(o, rd)])]
                k = keys([r.get("answer")] + [full[i][(o, rd)].get("answer") for o in peers])
                opp = sum(1 for x in k[1:] if x != k[0])
                fl[opp][0] += 1; fl[opp][1] += not nxt.get("correct")
    return {str(k): dict(cases=v[0], flipped=v[1], share=v[1] / v[0]) for k, v in sorted(fl.items())}


def section_six():
    full = _six_full()
    out = dict(n=len(full), A6=six_table(full, A6), W=six_table(full, W), W3=six_table(full, W3))
    out["W_allanswered"] = six_table(full, W, [i for i in full if all(present(full[i][(a, 0)]) for a in W)])
    out["W3_allanswered"] = six_table(full, W3, [i for i in full if all(present(full[i][(a, 0)]) for a in W3)])
    out["flips6"] = six_flips(full)
    return out


# ----------------------------------------------------------------------------- untraced (hand-derived in v8)
def untraced(deb, judge):
    c = deb["groups"]["contested"]; pj = deb["projection"]; w = judge["whole_733"]
    best = w["best_member_acc"]; orc = w["oracle"]
    return {
        # oracle over the four on the contested set at round 3: share of contested items where some answering agent is right
        "coverage_contested_r0": 1 - c["rounds"][0]["all_wrong"] / c["n"],
        "coverage_contested_r3": 1 - c["rounds"][3]["all_wrong"] / c["n"],
        # share of the best-member -> oracle gap collected by round-3 debate on the 733 (strongest-first projection)
        "gap_points": 100 * (orc - best),
        "collected_points": 100 * (pj["r3_nobreak"] - best),
        "collected_share": (pj["r3_nobreak"] - best) / (orc - best),
    }


def run():
    items, chosen, R0 = _data()
    deb, full, results, strong = section_debate(chosen, R0)
    syn = section_synth(chosen, R0, results, strong)
    jud = section_judge(chosen, R0, results, strong)
    out = dict(items=dict(n=len(items), groups={g: dict(n=sum(1 for r in items if r["group"] == g),
                                                        layers={l: sum(1 for r in items if r["group"] == g and r["layer"] == l) for l in HARD})
                                                for g in ("contested", "allwrong", "control", "allright")}),
               debate=deb, synth=syn, judge=jud, spend=section_spend(), six=section_six())
    out["untraced"] = untraced(deb, jud)
    # the same numbers in percent / points, as v8 prints them (no new computation)
    out["pct"] = dict(flip_vs3=100 * deb["flips"]["3"]["share"], six_flip_vs3=100 * out["six"]["flips6"]["3"]["share"],
                      six_W_gain_first_pts=100 * out["six"]["W"]["gain_first"])
    return {"debate_weak": out}


if __name__ == "__main__":
    import json
    from pathlib import Path
    r = run()
    for k, v in r.items(): (Path(__file__).resolve().parents[1] / "summaries" / "out2" / f"{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False))
    print(json.dumps(r["debate_weak"]["untraced"], indent=1))
