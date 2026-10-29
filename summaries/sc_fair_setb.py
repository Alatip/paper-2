"""Set B self-consistency against a fair starting point: the first of the five samples that gave an answer.
Offline, $0. python3 summaries/sc_fair_setb.py -> summaries/out2/sc_fair_setb.{json,md}

The paper's starting point is sample 1; for GLM 4.7 Flash sample 1 is often empty (no parsable answer or cut at the
length limit), which counts as an error, while the vote skips empty samples. Here the starting point is the first
sample with an answer (it may take 2-5 calls, so this comparison is about the vote itself, not about cost)."""
import csv, json, sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "operators"))
from p1stats.tests_stats import mcnemar_exact, holm
from p2parts.setb import plurality_codes

MODELS = {"glm": "z-ai/glm-4.7-flash", "deepseek": "deepseek/deepseek-v3.2"}
OUT = ROOT / "summaries/out2"
B = 10000  # bootstrap resamples, seed 0

S = {}
for r in csv.DictReader(open(ROOT / "datasets/set_b/answers.csv")):
    if r["op"] == "draw" and r["round"] == "0" and r["model"] in MODELS.values() and int(r["slot"]) <= 5:
        S.setdefault(r["model"], {}).setdefault(r["id"], {})[int(r["slot"])] = r

res = {}
for key, model in MODELS.items():
    items = sorted(S[model])
    s1, fa, mean_ans, vote, empty_first, none_all = [], [], [], [], 0, 0
    for i in items:
        rows = [S[model][i][k] for k in sorted(S[model][i])]
        codes = [r["code"] if (r["code"] and r["finish"] != "length") else "" for r in rows]
        oks = [int(r["correct"] or 0) for r in rows]
        answered = [j for j, c in enumerate(codes) if c]
        s1.append(oks[0])
        empty_first += not codes[0]
        if answered:
            fa.append(oks[answered[0]]); mean_ans.append(sum(oks[j] for j in answered) / len(answered))
        else:
            fa.append(0); mean_ans.append(0.0); none_all += 1
        vote.append(plurality_codes(codes, oks)[0])
    n = len(items)
    W = sum(v and not f for v, f in zip(vote, fa)); L = sum(f and not v for v, f in zip(vote, fa))
    rng = random.Random(0); bs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        bs.append(sum(vote[k] - fa[k] for k in idx) / n * 100)
    bs.sort()
    res[key] = {"model": model, "n": n, "acc_first": sum(s1) / n, "acc_first_answered": sum(fa) / n,
                "acc_mean_answered": sum(mean_ans) / n, "acc_vote": sum(vote) / n,
                "empty_first": empty_first, "none_answered": none_all,
                "gain_paper_pp": 100 * (sum(vote) - sum(s1)) / n,
                "gain_fair_pp": 100 * (sum(vote) - sum(fa)) / n, "fair_lo": bs[int(0.025 * B)],
                "fair_hi": bs[int(0.975 * B)], "W": W, "L": L, "p": mcnemar_exact(W, L),
                "start_shift_pp": 100 * (sum(fa) - sum(s1)) / n}

# Holm over the nine set-B comparisons (as in operators/setb_report.py) if GLM's comparison used this starting point
SB = json.loads((OUT / "setB.json").read_text())
FAM = [k for k in SB if k not in ("N", "single")]
ps = [res["glm"]["p"] if k == "sc.glm-4.7-flash" else SB[k]["mcnemar_p"] for k in FAM]
H = dict(zip(FAM, holm(ps)))
res["holm_if_replaced"] = {"family": FAM, "p_holm": H, "glm": H["sc.glm-4.7-flash"], "min": min(H.values()),
                           "n_significant": sum(v < 0.05 for v in H.values())}

# Minimal detectable effect (80 % power, normal approximation to the paired McNemar test) at n = 200, with the share of
# discordant problems observed for GLM vs its fair starting point; for one test (alpha 0.05) and for the Holm worst case
# (alpha 0.05 / 9). Delta = d * (2*pi - 1), pi = share of discordant problems won by the operator.
from statistics import NormalDist
Z = NormalDist().inv_cdf
def mde(n, d, alpha, power=0.8):
    D = n * d
    for k in range(1, 2001):
        pi = 0.5 + k / 4000
        if (2 * pi - 1) * D ** 0.5 >= Z(1 - alpha / 2) + Z(power) * 2 * (pi * (1 - pi)) ** 0.5:
            return 100 * d * (2 * pi - 1)
d_glm = (res["glm"]["W"] + res["glm"]["L"]) / res["glm"]["n"]
res["mde"] = {"n": res["glm"]["n"], "discordant_share": d_glm, "power": 0.8,
              "pp_alpha05": mde(res["glm"]["n"], d_glm, 0.05), "pp_holm9": mde(res["glm"]["n"], d_glm, 0.05 / 9)}
# the same calculation with the discordance of debate and synthesis against their starting point (the vote)
for k in ("debate7", "synth.opus"):
    d_k = (SB[k]["W"] + SB[k]["L"]) / SB[k]["N"]
    res["mde"][k] = {"discordant_share": d_k, "pp_alpha05": mde(SB[k]["N"], d_k, 0.05)}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "sc_fair_setb.json").write_text(json.dumps(res, indent=1))
md = ["| model | first sample | first sample with an answer | vote | vote vs first answered, pp [95 %] | W/L | p |",
      "|---|---|---|---|---|---|---|"]
for k, r in res.items():
    if k not in MODELS:
        continue
    md.append(f"| {r['model']} | {r['acc_first']:.3f} | {r['acc_first_answered']:.3f} | {r['acc_vote']:.3f} | "
              f"{r['gain_fair_pp']:+.1f} [{r['fair_lo']:+.1f}; {r['fair_hi']:+.1f}] | {r['W']}/{r['L']} | {r['p']:.3g} |")
(OUT / "sc_fair_setb.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
