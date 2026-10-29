"""Debate on the saturated pool: six different models, and six copies of DeepSeek V3.2 in three arms (paper v8 §6.3, table
6.1, restated in §6.7; tables B, B2, B3, B5, N and T of tech-note-5 results/tables.md).

Method (line-by-line port of tech-note-5 stats.section_debate / section_debate_deep / section_debate_offline /
section_homodebate and extra.section_quoted / section_fifth / section_ties, debate parts only):
- Mixed debate: 3 synchronous rounds; round 0 = each agent's temperature-0 matrix answer; agents in the fixed order
  DEBATE_AGENTS (DeepSeek, GLM, Mistral, Qwen, Haiku, Sonnet). Items = those with a row for every agent in rounds 0..3
  (225). A truncated row (correct is None) is a missing vote. Majority = plurality by answer identity (default paper-1 rule, p2parts/answer_id.py; v8 flag: canon, 7 significant
  digits) of the answering rows, ties to the first row in the fixed order. Fixed / broken = round-t majority right and
  round-0 wrong / the reverse. Unanimous = at least 2 answering rows and all equal.
- B2: mean pairwise φ (pairwise-complete, pairs with < 5 common rows dropped), mean tetrachoric ρ, ICC(1) on items with 6
  answers, n_eff = 6 / (1 + 5 φ̄), unanimous / unanimous & wrong / all answered wrong; flips: agent correct at t and
  answering at t+1, P(wrong at t+1) by the number of answering peers with a different answer.
- B3: per layer best single agent on its own answered items and the round-3 majority on the same items; stopping rule
  "stop at the first unanimous round, else round 3".
- T: three tie rules (fixed order; ties split as fractional credit; strongest agent first by round-0 accuracy with a
  truncated answer scoring 0, chosen in-sample); best agent per layer on its answered items.
- N: fixes on items with a truncated round-0 row; the 162 items all six answered at round 0; fixes that were round-0 ties;
  unanimous-wrong at round 3; whether the formerly truncated agent joins the winning round-3 majority.
- Homogeneous: six DeepSeek copies, round 0 = T=1 draws 0..5, arms plain / selfonly (no peers) / pressure (defer to the
  majority); each arm on its complete items and the three arms on their common items (230).
- untraced.strongest_fixed_broken (NEW): fixed / broken counts and exact McNemar p under the strongest-first tie rule (table
  6.1 row 3 prints 5 / 0 and p = 0.06 with no stored source line). McNemar p of the fixed-order row is also computed here.
Data: datasets/set_a/tn5-debate-mixed, datasets/set_a/tn5-matrix (round 0), datasets/set_a/tn5-debate-homo, datasets/set_a/tn5-samples. Offline, $0.
"""
import collections
from p2parts import tn5 as IO

A = IO.DEBATE_AGENTS


def _mj(full, i, rd, agents=A):
    return IO.maj_correct([full[i][(a, rd)] for a in agents])


def table_b(items=None):
    """Table B: per round agent acc, majority, unanimous, fixed, broken, truncated rows; 6 agents and 5 (no Mistral)."""
    items = items or IO.load_debate(); out = {}
    for agents, name in ((A, "6"), ([a for a in A if "mistral" not in a], "5")):
        full = [i for i, d in items.items() if all((a, rd) in d for a in agents for rd in range(4))]
        rows = {}; wrong0 = set()
        for rd in range(4):
            acc_n = acc_d = maj = una = fixed = broken = trunc = 0
            for i in full:
                rs = [items[i][(a, rd)] for a in agents]; pr = [r for r in rs if IO.present(r)]
                acc_n += sum(bool(r.get("correct")) for r in pr); acc_d += len(pr)
                trunc += sum(r.get("correct") is None for r in rs)
                ok = IO.maj_correct(rs); maj += ok
                top, cnt = IO.majority([r.get("answer") for r in pr])
                una += len(pr) >= 2 and cnt == len(pr)
                if rd == 0 and not ok: wrong0.add(i)
                if rd and ok and i in wrong0: fixed += 1
                if rd and not ok and i not in wrong0: broken += 1
            n = len(full)
            rows[str(rd)] = {"agent_acc": acc_n / max(acc_d, 1), "majority": maj / n, "unanimous": una / n, "fixed": fixed, "broken": broken, "truncated": trunc}
        out[name] = {"items": len(full), "rounds": rows}
    return out


def table_b2(full):
    out = {"items": len(full), "rounds": {}}
    for rd in range(4):
        cols = {a: [IO.cval(full[i][(a, rd)]) for i in full] for a in A}
        mp, _ = IO.pairwise_phi(cols); mt, nt = IO.pairwise_tetra(cols)
        mat = [[cols[a][j] for a in A] for j in range(len(full)) if all(cols[a][j] is not None for a in A)]
        una = unw = allw = 0
        for i in full:
            rs = [full[i][(a, rd)] for a in A]; pr = [r for r in rs if IO.present(r)]
            top, cnt = IO.majority([r.get("answer") for r in pr])
            if len(pr) >= 2 and cnt == len(pr):
                una += 1
                unw += (not bool(next(r for r in rs if IO.canon(r.get("answer")) == top).get("correct"))) if IO.AID.RULE == "v8" else (not IO.maj_correct(pr))
            allw += bool(pr) and not any(r.get("correct") for r in pr)
        out["rounds"][str(rd)] = {"phi": mp, "tetra": mt, "tetra_pairs": nt, "icc1": IO.icc1(mat), "n_eff": IO.n_eff(len(A), mp),
                                  "unanimous": una, "unanimous_wrong": unw, "all_wrong": allw, "items_6": len(mat)}
    fl = collections.defaultdict(lambda: [0, 0])
    for i in full:
        for rd in range(3):
            for a in A:
                r = full[i][(a, rd)]
                if not r.get("correct"): continue
                nxt = full[i][(a, rd + 1)]
                if not IO.present(nxt): continue
                peers = [o for o in A if o != a and IO.present(full[i][(o, rd)])]
                k = IO.vote_keys([r.get("answer")] + [full[i][(o, rd)].get("answer") for o in peers])
                opp = sum(1 for x in k[1:] if x != k[0])
                fl[opp][0] += 1; fl[opp][1] += not nxt.get("correct")
    out["flips"] = {str(k): {"cases": v[0], "flipped": v[1], "rate": v[1] / v[0]} for k, v in sorted(fl.items())}
    tk = collections.defaultdict(list)
    for i in full:
        for rd in range(1, 4):
            for a in A:
                t = (full[i][(a, rd)].get("tok") or {}).get("out")
                if t: tk[rd].append(t)
    out["tokens_per_turn"] = {str(rd): sum(v) / len(v) for rd, v in sorted(tk.items())}
    out["tokens_per_item_3_rounds"] = sum(sum(v) / len(v) for v in tk.values()) * 6
    return out


def table_b3(full, layers=IO.HARD):
    out = {"layers": {}}
    def mj(ids, rd): return sum(_mj(full, i, rd) for i in ids) / len(ids)
    for layer in layers:
        ids = [i for i in full if full[i][(A[0], 1)]["layer"] == layer]
        if len(ids) < 10: continue
        cols = {a: [IO.cval(full[i][(a, 0)]) for i in ids] for a in A}
        mp, _ = IO.pairwise_phi(cols)
        bestA = max(A, key=lambda a: sum(c for c in cols[a] if c) / max(1, sum(c is not None for c in cols[a])))
        best = sum(c for c in cols[bestA] if c) / max(1, sum(c is not None for c in cols[bestA]))
        bestA_ids = [i for i, c in zip(ids, cols[bestA]) if c is not None]
        best0 = max(sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in A)
        out["layers"][layer] = {"items": len(ids), "phi0": mp, "n_eff0": IO.n_eff(6, mp), "maj0": mj(ids, 0), "maj3": mj(ids, 3),
                                "best0_trunc0": best0, "gain_vs_best0": mj(ids, 3) - best0, "best_answered": best, "best_agent": bestA,
                                "best_answered_items": len(bestA_ids), "gain_vs_best_answered": mj(bestA_ids, 3) - best,
                                "gain_vs_maj0": mj(ids, 3) - mj(ids, 0)}
    ids = list(full); acc = rounds_used = 0; by_stop = collections.Counter(); wrong_by_stop = collections.Counter()
    for i in ids:
        for rd in range(4):
            rs = [full[i][(a, rd)] for a in A]; top, cnt = IO.majority([r.get("answer") for r in rs if IO.present(r)]); npr = sum(IO.present(r) for r in rs)
            if (npr >= 2 and cnt == npr) or rd == 3:
                ok = IO.maj_correct(rs); acc += ok; rounds_used += rd; by_stop[rd] += 1; wrong_by_stop[rd] += not ok; break
    out["stopping"] = {"accuracy": acc / len(ids), "mean_rounds": rounds_used / len(ids),
                       "stops": {str(rd): {"items": by_stop[rd], "wrong": wrong_by_stop[rd]} for rd in sorted(by_stop)}}
    return out


def _mj_rows(rows):
    rows = [r for r in rows if IO.present(r)]
    if IO.AID.RULE != "v8": return float(IO.maj_correct(rows))
    top, _ = IO.majority([r.get("answer") for r in rows])
    if top is None: return 0.0
    return float(bool(next((r for r in rows if IO.canon(r.get("answer")) == top), {}).get("correct")))


def _mj_split(rows):
    return IO.split_correct([r for r in rows if IO.present(r)])


def table_t(full, layers=IO.HARD):
    ids = sorted(full)
    acc0 = {a: sum(bool(full[i][(a, 0)].get("correct")) for i in ids) / len(ids) for a in A}
    strong = sorted(A, key=lambda a: -acc0[a])
    rules = {"fixed": lambda rows: _mj_rows(rows),
             "split": _mj_split,
             "strongest": lambda rows: _mj_rows([{r["agent"]: r for r in rows}[a] for a in strong])}
    out = {"strongest_order": strong, "rules": {}}
    for name, f in rules.items():
        def M(sub, rd): return sum(f([dict(full[i][(a, rd)], agent=a) for a in A]) for i in sub) / len(sub)
        d = {"maj0": M(ids, 0), "maj3": M(ids, 3), "gain": M(ids, 3) - M(ids, 0), "layers": {}}
        for layer in layers:
            li = [i for i in ids if full[i][(A[0], 1)]["layer"] == layer]
            if li: d["layers"][layer] = {"maj0": M(li, 0), "maj3": M(li, 3)}
        if name != "split":   # NEW: fixed / broken and McNemar under this rule (table 6.1 columns)
            fx = sum(1 for i in ids if not f([dict(full[i][(a, 0)], agent=a) for a in A]) and f([dict(full[i][(a, 3)], agent=a) for a in A]))
            br = sum(1 for i in ids if f([dict(full[i][(a, 0)], agent=a) for a in A]) and not f([dict(full[i][(a, 3)], agent=a) for a in A]))
            d.update({"fixed": fx, "broken": br, "mcnemar_p": IO.mcnemar(fx, br)})
        out["rules"][name] = d
    out["best_answered"] = {}
    for layer in layers:
        li = [i for i in ids if full[i][(A[0], 1)]["layer"] == layer]
        if not li: continue
        best = None
        for a in A:
            ans = [i for i in li if IO.present(full[i][(a, 0)])]
            acc = sum(bool(full[i][(a, 0)]["correct"]) for i in ans) / len(ans)
            if best is None or acc > best[0]: best = (acc, a, ans)
        acc, a, ans = best
        m0 = sum(_mj_rows([full[i][(b, 0)] for b in A]) for i in ans) / len(ans); m3 = sum(_mj_rows([full[i][(b, 3)] for b in A]) for i in ans) / len(ans)
        out["best_answered"][layer] = {"agent": a, "answered": len(ans), "items": len(li), "acc": acc, "maj0": m0, "maj3": m3, "maj3_minus_agent": m3 - acc,
                                       "maj3_all": sum(_mj_rows([full[i][(b, 3)] for b in A]) for i in li) / len(li)}
    return out


def quoted(full):
    """Table N lines on the mixed debate (extra.section_quoted / section_fifth)."""
    fixes = [i for i in full if not _mj(full, i, 0) and _mj(full, i, 3)]
    cut = [i for i in fixes if any(not IO.present(full[i][(a, 0)]) for a in A)]
    six = [i for i in full if all(IO.present(full[i][(a, 0)]) for a in A)]
    m0 = sum(_mj(full, i, 0) for i in six) / len(six); m3 = sum(_mj(full, i, 3) for i in six) / len(six)
    tied = []; split = 0.0
    for i in full:
        rs = [full[i][(a, 0)] for a in A if IO.present(full[i][(a, 0)])]
        kk = [IO.canon(r.get("answer")) for r in rs] if IO.AID.RULE == "v8" else IO.vote_keys([r.get("answer") for r in rs])
        cnt = collections.Counter(kk)
        if not cnt: continue
        top = max(cnt.values()); tops = [k for k, v in cnt.items() if v == top]
        corr = {k for k, r in zip(kk, rs) if r.get("correct")}
        split += len([k for k in tops if k in corr]) / len(tops)
        if i in fixes and len(tops) > 1 and any(k in corr for k in tops): tied.append(i)
    uw = []
    for i in full:
        rs = [full[i][(a, 3)] for a in A if IO.present(full[i][(a, 3)])]
        top, cnt = IO.majority([r.get("answer") for r in rs])
        if len(rs) >= 2 and cnt == len(rs) and not any(r.get("correct") for r in rs): uw.append(i)
    joined = 0
    for i in cut:
        ag = [a for a in A if IO.present(full[i][(a, 3)])]
        top, _ = IO.majority([full[i][(a, 3)].get("answer") for a in ag])
        kk = dict(zip(ag, IO.vote_keys([full[i][(a, 3)].get("answer") for a in ag])))
        joined += any(not IO.present(full[i][(a, 0)]) and a in kk and kk[a] == top for a in A)
    w_cut = [i for i in full if not _mj(full, i, 0) and any(not IO.present(full[i][(a, 0)]) for a in A)]
    w_six = [i for i in full if not _mj(full, i, 0) and all(IO.present(full[i][(a, 0)]) for a in A)]
    return {"items": len(full), "fixes": len(fixes), "fixes_cut": len(cut), "fixes_cut_joined": joined, "fixes_cut_not_joined": len(cut) - joined,
            "six_items": len(six), "six_maj0": m0, "six_maj3": m3, "six_fixes": sum(1 for i in six if not _mj(full, i, 0) and _mj(full, i, 3)),
            "fixes_tied": len(tied), "split_maj0": split / len(full), "unanimous_wrong_r3": len(uw), "unanimous_wrong_r3_ids": sorted(uw),
            "wrong0_cut": len(w_cut), "wrong0_cut_fixed": sum(_mj(full, i, 3) for i in w_cut), "wrong0_six": len(w_six), "wrong0_six_fixed": sum(_mj(full, i, 3) for i in w_six)}


def homogeneous(layers=IO.HARD):
    C = [f"deepseek/deepseek-v3.2#{i}" for i in range(6)]
    out = {"arms": {}}; arms_ = {}; common = None
    for arm in IO.ARMS:
        items = IO.load_homo(arm, layers)
        full = [i for i, dd in items.items() if all((a, rd) in dd for a in C for rd in range(4))]
        arms_[arm] = items; common = set(full) if common is None else common & set(full)

        def stat(rd):
            acc_n = acc_d = maj = una = unw = allw = 0
            for i in full:
                rs = [items[i][(a, rd)] for a in C]; pr = [r for r in rs if IO.present(r)]
                acc_n += sum(bool(r.get("correct")) for r in pr); acc_d += len(pr); ok = IO.maj_correct(rs); maj += ok
                top, cnt = IO.majority([r.get("answer") for r in pr]); u = len(pr) >= 2 and cnt == len(pr); una += u; unw += (u and not ok); allw += bool(pr) and not any(r.get("correct") for r in pr)
            cols = {a: [IO.cval(items[i][(a, rd)]) for i in full] for a in C}
            mp, _ = IO.pairwise_phi(cols)
            return {"agent_acc": acc_n / max(acc_d, 1), "majority": maj / len(full), "unanimous": una, "unanimous_wrong": unw, "all_wrong": allw, "phi": mp}
        r0, r3 = stat(0), stat(3)
        fixed = sum(1 for i in full if not IO.maj_correct([items[i][(a, 0)] for a in C]) and IO.maj_correct([items[i][(a, 3)] for a in C]))
        broken = sum(1 for i in full if IO.maj_correct([items[i][(a, 0)] for a in C]) and not IO.maj_correct([items[i][(a, 3)] for a in C]))
        nrows = sum(len(IO.homo(arm, L)) for L in layers); trows = sum(r.get("correct") is None for L in layers for r in IO.homo(arm, L))
        out["arms"][arm] = {"items": len(full), "r0": r0, "r3": r3, "fixed": fixed, "broken": broken, "rows_r1_3": nrows, "truncated_rows_r1_3": trows}
    m0 = sum(IO.maj_correct([arms_["plain"][i][(a, 0)] for a in C]) for i in common) / len(common)
    cm = {"items": len(common), "maj0": m0}
    for arm, items in arms_.items():
        m3 = sum(IO.maj_correct([items[i][(a, 3)] for a in C]) for i in common) / len(common)
        fx = sum(1 for i in common if not IO.maj_correct([items[i][(a, 0)] for a in C]) and IO.maj_correct([items[i][(a, 3)] for a in C]))
        br = sum(1 for i in common if IO.maj_correct([items[i][(a, 0)] for a in C]) and not IO.maj_correct([items[i][(a, 3)] for a in C]))
        cm[arm] = {"maj3": m3, "gain": m3 - m0, "fixed": fx, "broken": br}
    cm["selfonly_share_of_plain_gain"] = cm["selfonly"]["gain"] / cm["plain"]["gain"]
    out["common"] = cm
    return out


def run() -> dict:
    items = IO.load_debate(); full = {i: d for i, d in items.items() if all((a, rd) in d for a in A for rd in range(4))}
    b = table_b(items); t = table_t(full)
    mixed = {"B": b, "B2": table_b2(full), "B3": table_b3(full), "T": t, "N": quoted(full),
             "fixed_order_mcnemar_p": IO.mcnemar(b["6"]["rounds"]["3"]["fixed"], b["6"]["rounds"]["3"]["broken"])}
    ba = t["best_answered"].values()
    mixed["summary"] = {"layer_items_min": min(r["items"] for r in ba), "layer_items_max": max(r["items"] for r in ba),
                        "vs_best_min": min(r["maj3_minus_agent"] for r in ba), "vs_best_max": max(r["maj3_minus_agent"] for r in ba)}
    return {"debate": {"mixed": mixed, "homogeneous": homogeneous(),
                       "untraced": {"strongest_fixed_broken": {k: t["rules"]["strongest"][k] for k in ("fixed", "broken", "mcnemar_p")}}}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1))
