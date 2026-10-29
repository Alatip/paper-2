"""Set B: how the answers behave (paper 2, Section 4.1.4). Offline, $0.
python3 summaries/mechanics_setb.py -> summaries/out2/mechanics_setb.{json,md}

Q1  scatter of one model's samples: per-problem number of correct samples vs a common-rate binomial, wrong-answer
    attractors, number of distinct answer codes, output length of right / wrong samples.
Q2  ties and notation splits in the votes: how many vote-vs-starting-point changes are decided by the tie rule, how many
    problems have correct answers split over several codes, and the soft rule (all correct answers = one group).
Q3  debate dynamics: member accuracy, vote, distinct codes and switches by round; correct minority wins and losses.
R   robustness of the seven-model round-0 vote (starting point of debate and synthesis) to the tie rule: tie counted
    as the expected value of a random tie-break, and every one of the 7! = 5040 pool orders (exact enumeration), with
    the comparisons that use this vote (vs Claude Opus 4.6 in-sample, vs the held-out best model, debate and synthesis
    vs the vote). Debate transcripts and the synthesis answer are fixed: only the counting rule changes.

Conventions as operators/setb_report.py: answer code '' = no answer (incl. answers cut at the length limit);
plurality via p2parts.setb.plurality_codes (tie -> first member in pool order)."""
import csv, json, sys, math, itertools, collections, statistics as st
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "operators"))
from p1stats.tests_stats import mcnemar_exact, holm
from p2parts import setb as SB
from p2parts.setb import plurality_codes

D = ROOT / "datasets/set_b"
OUT = ROOT / "summaries/out2"
POOL = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "deepseek/deepseek-v3.2", "z-ai/glm-4.7-flash",
        "qwen/qwen3.8-27b", "moonshotai/kimi-k2.5"]
DS, GLM, OPUS = POOL[3], POOL[4], POOL[2]
SHORT = {DS: "deepseek", GLM: "glm", OPUS: "opus"}
ITEMS = list(csv.DictReader(open(D / "items.csv")))
IT = {r["id"]: r for r in ITEMS}
IDS = [r["id"] for r in ITEMS]
N = len(IDS)
A = {}
for r in csv.DictReader(open(D / "answers.csv")):
    r["correct"] = int(r["correct"]); r["tok_out"] = int(r["tok_out"] or 0)
    A[(r["id"], r["op"], r["model"], int(r["slot"]), int(r["round"]))] = r
get = lambda i, op, m, s=1, k=0: A.get((i, op, m, s, k))
ok = lambda r: r["correct"] if r else 0
cd = lambda r: r["code"] if r else ""


def soft_keys(rs):
    return [("__OK__" if ok(r) else cd(r)) if (ok(r) or cd(r)) else "" for r in rs]


def vote(rs, soft=False):
    """(correct, tie at the top, winning key) under the paper's rule (tie -> first member in the given order)."""
    keys = soft_keys(rs) if soft else [cd(r) for r in rs]
    c, top = plurality_codes(keys, [ok(r) for r in rs])
    cnt = collections.Counter(k for k in keys if k)
    tie = top is not None and sum(1 for k in cnt if cnt[k] == cnt[top]) > 1
    return c, tie, top


def vote_frac(rs):
    """Expected correctness under a uniformly random choice among the tied top groups (= mean over all member orders,
    since tied groups have equal size); equals 1/2 for a two-way tie between a right and a wrong group."""
    keys = [cd(r) for r in rs]
    cnt = collections.Counter(k for k in keys if k)
    if not cnt:
        return 0.0
    mx = max(cnt.values()); tops = [k for k in cnt if cnt[k] == mx]
    return sum(any(ok(r) for r in rs if cd(r) == k) for k in tops) / len(tops)


def split(rs):  # correct answers of the problem spread over more than one code
    return len({cd(r) for r in rs if ok(r) and cd(r)}) > 1


res = {"N": N}

# ---------------- Q1: scatter of one model's samples ----------------
q1 = {}
for m, n in ((DS, 5), (GLM, 5), (OPUS, 3)):
    kd = collections.Counter(); tier = collections.defaultdict(list); ndist = collections.Counter()
    attract = collections.Counter(); mixed_w = []; tok = collections.defaultdict(list)
    for i in IDS:
        rs = [get(i, "draw", m, s) for s in range(1, n + 1)]
        k = sum(ok(r) for r in rs); kd[k] += 1; tier[IT[i]["tier"]].append(k)
        ndist[len({cd(r) for r in rs if cd(r)})] += 1
        wc = collections.Counter(cd(r) for r in rs if cd(r) and not ok(r)); nw = sum(wc.values())
        if nw >= 2:
            attract["shared" if wc.most_common(1)[0][1] >= 2 else "scattered"] += 1
            if 0 < k < n:
                mixed_w.append(nw == wc.most_common(1)[0][1])
        for r in rs:
            t = "cut" if r["finish"] == "length" else ("noans" if not cd(r) else ("right" if ok(r) else "wrong"))
            tok[t].append(r["tok_out"])
    pbar = sum(kd[k] * k for k in kd) / (N * n)
    exp = {k: N * math.comb(n, k) * pbar ** k * (1 - pbar) ** (n - k) for k in range(n + 1)}
    paired = []
    for i in IDS:
        rs = [get(i, "draw", m, s) for s in range(1, n + 1)]
        R = [r["tok_out"] for r in rs if ok(r)]; W = [r["tok_out"] for r in rs if cd(r) and not ok(r)]
        if R and W:
            paired.append(st.median(W) - st.median(R))
    q1[SHORT[m]] = {
        "n_samples": n, "k_dist": {str(k): kd[k] for k in range(n + 1)}, "p_mean": pbar,
        "k_binom": {str(k): exp[k] for k in range(n + 1)},
        "all_right": kd[n], "all_wrong": kd[0], "mixed": N - kd[n] - kd[0],
        "binom_all_right": exp[n], "binom_all_wrong": exp[0], "binom_mixed": N - exp[n] - exp[0],
        "tier": {t: {"items": len(v), "acc": sum(v) / len(v) / n, "mixed": sum(1 for k in v if 0 < k < n),
                     "all_wrong": sum(1 for k in v if k == 0), "all_right": sum(1 for k in v if k == n)}
                 for t, v in sorted(tier.items())},
        "distinct_codes": {str(k): ndist[k] for k in sorted(ndist)},
        "one_code": ndist[1], "three_plus_codes": sum(v for k, v in ndist.items() if k >= 3),
        "wrong2": attract["shared"] + attract["scattered"], "attractor": attract["shared"],
        "scattered": attract["scattered"],
        "attractor_share": attract["shared"] / max(1, attract["shared"] + attract["scattered"]),
        "mixed_wrong2": len(mixed_w), "mixed_one_wrong_code": sum(mixed_w),
        "tok": {t: {"n": len(v), "median": int(st.median(v))} for t, v in tok.items() if v},
        "paired_items": len(paired), "paired_wrong_longer": sum(x > 0 for x in paired),
        "paired_median_diff": float(st.median(paired)) if paired else None,
    }
# where the scatter is: DeepSeek state x GLM state (0 = all wrong, 5 = all right, mix)
st3 = lambda k, n: "0" if k == 0 else (str(n) if k == n else "mix")
cross = collections.Counter()
for i in IDS:
    a = sum(ok(get(i, "draw", DS, s)) for s in range(1, 6)); b = sum(ok(get(i, "draw", GLM, s)) for s in range(1, 6))
    cross[f"{st3(a, 5)}|{st3(b, 5)}"] += 1
q1["cross_ds_glm"] = dict(cross)
q1["ds_mixed_glm_any"] = {s: cross.get(f"mix|{s}", 0) for s in ("0", "mix", "5")}
q1["ds_all_right_glm_any"] = {s: cross.get(f"5|{s}", 0) for s in ("0", "mix", "5")}
res["q1"] = q1

# ---------------- Q2: ties and notation splits ----------------
def first_answered(m, n):
    def f(i):
        for s in range(1, n + 1):
            r = get(i, "draw", m, s)
            if r and cd(r):
                return ok(r)
        return 0
    return f


def gain(base, voters):
    W = L = Wt = Lt = sw = sl = splits = flips = 0; tie_all_distinct = tie_all_distinct_answered = 0
    tie_first = 0
    for i in IDS:
        b = base(i); rs = voters(i); v, tie, top = vote(rs); vs = vote(rs, True)[0]
        if split(rs):
            splits += 1; flips += v != vs
        if v != b:
            if v > b: W += 1; Wt += tie
            else: L += 1; Lt += tie
            if tie:
                codes = [cd(r) for r in rs if cd(r)]
                if len(set(codes)) == len(codes):
                    tie_all_distinct_answered += 1
                    if len(codes) == len(rs):
                        tie_all_distinct += 1
                        tie_first += cd(rs[0]) == top
        sw += vs > b; sl += vs < b
    return {"W": W, "L": L, "net": W - L, "W_tie": Wt, "L_tie": Lt, "changes": W + L, "tie_changes": Wt + Lt,
            "splits": splits, "split_flips": flips, "W_soft": sw, "L_soft": sl, "net_soft": sw - sl,
            "tie_all_distinct": tie_all_distinct, "tie_all_distinct_answered": tie_all_distinct_answered,
            "tie_all_distinct_first_member": tie_first}


q2 = {}
for m, n in ((DS, 5), (GLM, 5), (OPUS, 3)):
    vt = lambda i, m=m, n=n: [get(i, "draw", m, s) for s in range(1, n + 1)]
    q2[f"sc_{SHORT[m]}_vs_first"] = gain(lambda i, m=m: ok(get(i, "draw", m)), vt)
    q2[f"sc_{SHORT[m]}_vs_first_answered"] = gain(first_answered(m, n), vt)
v7 = lambda i: [get(i, "draw", m) for m in POOL]
q2["vote7_vs_opus"] = gain(lambda i: ok(get(i, "draw", OPUS)), v7)
res["q2"] = q2

# ---------------- Q3: debate dynamics ----------------
def dyn(items, rounds):
    out = {"items": len(items), "rounds": []}
    for f in rounds:
        nm = sum(len(f(i)) for i in items)
        out["rounds"].append({
            "member_acc": sum(ok(r) for i in items for r in f(i)) / nm,
            "vote_correct": sum(vote(f(i))[0] for i in items),
            "mean_codes": sum(len({cd(r) for r in f(i) if cd(r)}) for i in items) / len(items),
            "unanimous": sum(1 for i in items if len({cd(r) for r in f(i)}) == 1 and cd(f(i)[0])),
            "no_answer": sum(1 for i in items for r in f(i) if not cd(r))})
    out["switches"] = []
    for a, b in ((0, 1), (1, 2)):
        sw = wr = rw = stay = to0 = 0
        for i in items:
            r0 = vote(rounds[0](i))[2]
            for x, y in zip(rounds[a](i), rounds[b](i)):
                if cd(x) != cd(y):
                    sw += 1; wr += ok(x) < ok(y); rw += ok(x) > ok(y); to0 += bool(cd(y)) and cd(y) == r0
                else:
                    stay += 1
        out["switches"].append({"from": a, "to": b, "switches": sw, "members": sw + stay, "share": sw / (sw + stay),
                                "wrong_to_right": wr, "right_to_wrong": rw, "to_r0_winner": to0})
    fin = rounds[-1]
    win = [i for i in items if vote(rounds[0](i))[0] == 0 and vote(fin(i))[0] == 1 and any(ok(r) for r in rounds[0](i))]
    lost = [i for i in items if vote(rounds[0](i))[0] == 1 and vote(fin(i))[0] == 0]
    wrongc = [i for i in items if len({cd(r) for r in fin(i)}) == 1 and cd(fin(i)[0]) and not ok(fin(i)[0])]
    anyR = [i for i in items if any(ok(r) for r in rounds[0](i))]
    out.update({
        "same_winner_as_r0": sum(1 for i in items if vote(fin(i))[2] == vote(rounds[0](i))[2]),
        "minority_wins": len(win), "minority_wins_right_r0": [sum(ok(r) for r in rounds[0](i)) for i in win],
        "majority_lost": len(lost), "majority_lost_right_r0": [sum(ok(r) for r in rounds[0](i)) for i in lost],
        "wrong_unanimous": len(wrongc),
        "wrong_unanimous_none_right_r0": sum(1 for i in wrongc if not any(ok(r) for r in rounds[0](i))),
        "any_right_r0": len(anyR), "any_right_r0_final_right": sum(vote(fin(i))[0] for i in anyR)})
    return out


it7 = [i for i in IDS if get(i, "debate7", POOL[0], 1, 1)]
d7 = dyn(it7, [lambda i: [get(i, "draw", m) for m in POOL]] +
         [lambda i, k=k: [get(i, "debate7", m, 1, k) for m in POOL] for k in (1, 2)])
d7["per_model"] = {m: [sum(ok(get(i, "draw", m) if k == 0 else get(i, "debate7", m, 1, k)) for i in it7)
                       for k in (0, 1, 2)] for m in POOL}
itc = [i for i in IDS if get(i, "copies", DS, 1, 1)]
cp = dyn(itc, [lambda i: [get(i, "draw", DS, s) for s in range(1, 6)]] +
         [lambda i, k=k: [get(i, "copies", DS, s, k) for s in range(1, 6)] for k in (1, 2)])
for x in (d7, cp):
    x["vote_gain"] = x["rounds"][2]["vote_correct"] - x["rounds"][0]["vote_correct"]
    x["member_gain_pp"] = 100 * (x["rounds"][2]["member_acc"] - x["rounds"][0]["member_acc"])
    x["wrong_unanimous_share"] = x["wrong_unanimous"] / x["items"]
res["q3"] = {"debate7": d7, "copies": cp}

# ---------------- R: robustness of the seven-model round-0 vote ----------------
r0 = {i: [get(i, "draw", m) for m in POOL] for i in IDS}
r2 = {i: [get(i, "debate7", m, 1, 2) for m in POOL] for i in IDS}
deb = {i: all(r2[i]) for i in IDS}
synth = np.array([ok(get(i, "synth", OPUS)) for i in IDS], float)
opus1 = np.array([ok(get(i, "draw", OPUS)) for i in IDS], float)
opus3 = np.array([vote([get(i, "draw", OPUS, s) for s in (1, 2, 3)])[0] for i in IDS], float)   # Opus SC@3 (fixed)
SBJ = json.loads((OUT / "setB.json").read_text())
FAM = [k for k in SBJ if k not in ("N", "single")]   # the nine-comparison Holm family of setb_report.py
P = np.array([[ok(r) for r in r0[i]] for i in IDS], float)
hb = SB.heldout_best(P, SB.splits([f"{IT[i]['tier']}|{IT[i]['area']}" for i in IDS]))   # as setb_report.py
HB = float(hb.mean())


def vectors(order):
    v = np.array([plurality_codes([cd(r0[i][j]) for j in order], [ok(r0[i][j]) for j in order])[0] for i in IDS], float)
    d = np.array([plurality_codes([cd(r2[i][j]) for j in order], [ok(r2[i][j]) for j in order])[0] if deb[i] else v[k]
                  for k, i in enumerate(IDS)], float)
    return v, d


def comps(v, d):
    def wl(a, b):  # b vs a
        W = int(((a == 0) & (b == 1)).sum()); L = int(((a == 1) & (b == 0)).sum()); return W, L
    out = {"vote": v.mean(), "debate": d.mean(), "vote_vs_opus_pp": 100 * (v.mean() - opus1.mean()),
           "vote_vs_heldout_pp": 100 * (v.mean() - HB), "debate_vs_heldout_pp": 100 * (d.mean() - HB), "debate_vs_vote_pp": 100 * (d.mean() - v.mean()),
           "synth_vs_vote_pp": 100 * (synth.mean() - v.mean()), "vote_vs_opus3_pp": 100 * (v.mean() - opus3.mean())}
    if set(np.unique(np.r_[v, d])) <= {0.0, 1.0}:
        for name, a, b in (("vote_vs_opus", opus1, v), ("debate_vs_vote", v, d), ("synth_vs_vote", v, synth)):
            W, L = wl(a, b); out[name + "_W"] = W; out[name + "_L"] = L; out[name + "_p"] = mcnemar_exact(W, L)
        W, L = wl(opus3, v); out["vote_vs_opus3_p"] = mcnemar_exact(W, L)
        rep = {"debate7": out["debate_vs_vote_p"], "synth.opus": out["synth_vs_vote_p"]}
        H = dict(zip(FAM, holm([rep.get(k, SBJ[k]["mcnemar_p"]) for k in FAM])))
        out["debate_holm"] = H["debate7"]; out["synth_holm"] = H["synth.opus"]
    return out


base = comps(*vectors(list(range(7))))
vf = np.array([vote_frac(r0[i]) for i in IDS])
df = np.array([vote_frac(r2[i]) if deb[i] else vf[k] for k, i in enumerate(IDS)])
half = comps(vf, df)
tie_items_v = sum(1 for i in IDS if vote(r0[i])[1])
tie_items_v_mixed = sum(1 for i, x in zip(IDS, vf) if 0 < x < 1)
tie_items_d = sum(1 for k, i in enumerate(IDS) if deb[i] and 0 < vote_frac(r2[i]) < 1)

# exact enumeration of all pool orders; only problems with a tie at the top can change, so cache the rest
perm_rows = []
for order in itertools.permutations(range(7)):
    perm_rows.append(comps(*vectors(list(order))))
keys = ["vote", "debate", "vote_vs_opus_pp", "vote_vs_heldout_pp", "debate_vs_vote_pp", "synth_vs_vote_pp",
        "vote_vs_opus3_pp", "vote_vs_opus_p", "debate_vs_vote_p", "synth_vs_vote_p", "vote_vs_opus3_p", "debate_holm",
        "synth_holm"]
perm = {"n_orders": len(perm_rows)}
for k in keys:
    x = np.array([r[k] for r in perm_rows])
    perm[k] = {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
               "p2_5": float(np.percentile(x, 2.5)), "p97_5": float(np.percentile(x, 97.5))}
for k in ("vote_vs_opus_p", "debate_vs_vote_p", "synth_vs_vote_p", "vote_vs_opus3_p", "debate_holm", "synth_holm"):
    perm[k]["share_below_05"] = float(np.mean([r[k] < 0.05 for r in perm_rows]))
perm["vote_rank_of_paper_order"] = float(np.mean([r["vote"] < base["vote"] for r in perm_rows]))   # share of orders below
for k in ("vote", "debate"):
    vals = [r[k] for r in perm_rows]
    perm[k]["values"] = {f"{x:.3f}": vals.count(x) for x in sorted(set(vals))}
res["robust"] = {"opus3": float(opus3.mean()), "paper_order": base, "tie_half": half, "heldout_best": HB, "opus_first": float(opus1.mean()),
                 "synth": float(synth.mean()), "tie_items_vote": tie_items_v, "tie_items_vote_mixed": tie_items_v_mixed,
                 "tie_items_debate_mixed": tie_items_d, "orders": perm,
                 "half_minus_paper_pp": 100 * (half["vote"] - base["vote"])}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "mechanics_setb.json").write_text(json.dumps(res, indent=1, default=float))

# ---------------- markdown ----------------
L = ["# Set B: how the answers behave (Section 4.1.4)", "", "Generated by `summaries/mechanics_setb.py`.", "",
     "## Q1 samples of one model", "", "| model | samples | all right | mixed | all wrong | binomial: all right / mixed / all wrong | one code | attractor / scattered | median tokens right / wrong |",
     "|---|---|---|---|---|---|---|---|---|"]
for m in ("deepseek", "glm", "opus"):
    x = q1[m]
    L.append(f"| {m} | {x['n_samples']} | {x['all_right']} | {x['mixed']} | {x['all_wrong']} | {x['binom_all_right']:.1f} / "
             f"{x['binom_mixed']:.1f} / {x['binom_all_wrong']:.1f} | {x['one_code']} | {x['attractor']} / {x['scattered']} | "
             f"{x['tok']['right']['median']} / {x['tok']['wrong']['median']} |")
L += ["", f"DeepSeek x GLM states: {q1['cross_ds_glm']}", "", "## Q2 ties and notation splits", "",
      "| comparison | W/L | net | W by tie | L by tie | splits | split flips | W/L soft | net soft |", "|---|---|---|---|---|---|---|---|---|"]
for k, x in q2.items():
    L.append(f"| {k} | {x['W']}/{x['L']} | {x['net']:+d} | {x['W_tie']} | {x['L_tie']} | {x['splits']} | {x['split_flips']} | "
             f"{x['W_soft']}/{x['L_soft']} | {x['net_soft']:+d} |")
L += ["", "## Q3 debate rounds", ""]
for name, x in (("debate7", d7), ("copies", cp)):
    L += [f"### {name}: {x['items']} debated problems", "", "| round | member acc | vote correct | mean codes | unanimous | no answer |",
          "|---|---|---|---|---|---|"]
    for k, r in enumerate(x["rounds"]):
        L.append(f"| {k} | {r['member_acc']:.3f} | {r['vote_correct']} | {r['mean_codes']:.2f} | {r['unanimous']} | {r['no_answer']} |")
    L += ["", f"minority wins {x['minority_wins']}, majority lost {x['majority_lost']}, wrong unanimous {x['wrong_unanimous']}", ""]
b, h, p = base, half, perm
L += ["## R robustness of the seven-model vote", "", "| rule | vote | debate | vote - Opus | vote - held-out best | debate - vote | synthesis - vote |",
      "|---|---|---|---|---|---|---|",
      f"| paper order | {b['vote']:.3f} | {b['debate']:.3f} | {b['vote_vs_opus_pp']:+.1f} | {b['vote_vs_heldout_pp']:+.1f} | {b['debate_vs_vote_pp']:+.1f} | {b['synth_vs_vote_pp']:+.1f} |",
      f"| tie = expected value | {h['vote']:.3f} | {h['debate']:.3f} | {h['vote_vs_opus_pp']:+.1f} | {h['vote_vs_heldout_pp']:+.1f} | {h['debate_vs_vote_pp']:+.1f} | {h['synth_vs_vote_pp']:+.1f} |"]
for s in ("mean", "min", "max", "p2_5", "p97_5"):
    L.append(f"| orders {s} | {p['vote'][s]:.3f} | {p['debate'][s]:.3f} | {p['vote_vs_opus_pp'][s]:+.1f} | {p['vote_vs_heldout_pp'][s]:+.1f} | "
             f"{p['debate_vs_vote_pp'][s]:+.1f} | {p['synth_vs_vote_pp'][s]:+.1f} |")
L += ["", f"p over orders: vote vs Opus {p['vote_vs_opus_p']['min']:.3g}-{p['vote_vs_opus_p']['max']:.3g}; "
      f"debate vs vote {p['debate_vs_vote_p']['min']:.3g}-{p['debate_vs_vote_p']['max']:.3g}; "
      f"synthesis vs vote {p['synth_vs_vote_p']['min']:.3g}-{p['synth_vs_vote_p']['max']:.3g}"]
(OUT / "mechanics_setb.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
