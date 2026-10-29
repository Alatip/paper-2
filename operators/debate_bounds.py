"""Bounds on debate (v8 §6.6; restated in §6.7): the Moore–Shannon restoring-organ curve (axis H, H3) and the MAP ceiling of a
homogeneous DeepSeek debate (H6c).

Method (ported from the H3 and H6c analysis scripts, not published; their helpers are copied below. Those scripts read
the live corpus-traces rows: no label audit, answers compared as raw strings):
  * H3: the six-agent mixed debate (round 0 = matrix rows, rounds 1-3 debate rows), items complete in all 4 rounds for all 6
    agents (229). c = number of agents correct at round 0 (a cut-off row counts as not correct); majority correct at round r =
    plurality of the non-empty raw answers (Counter.most_common, ties to the first answer seen in the fixed agent order) is right.
    Wilson 95 % on the round-3 rate. Crossover = some c with r3 rate below r0 rate and the Wilson upper bound below r0.
    Share of items with >= 4 of 6 correct at round 0 = the "79 %".
  * H6c: DeepSeek's MAP ceiling = share of items where the mode of its own 16 T = 1 samples (raw answer strings) is right.
    Round-3 majority of the homogeneous six-copy debate (arms plain / selfonly / pressure; >= 5 copies with a round-3 row) and
    of the mixed six-agent debate (all 6 agents at round 3), each minus the MAP indicator on the same items; percentile
    bootstrap (1000 resamples of items, random.Random(0) consumed in the original order: layers mmlu-pro, gpqa-diamond,
    gsm-hard, olympiad, pooled; within a layer the arms, then mixed); interval = sorted means [25], [974].
Answer identity (decision 2026-10-18): default = paper 1's rule (p2parts/answer_id.py: 12-digit canon + decimal rounding,
chained groups per vote; a group is correct if any member is); P2_ANSWER_RULE=v8 restores the raw-string identity with the
first row of the winning string deciding (reproduces v8). Ties: first answer seen in the fixed order (both rules).
Implementation decision: none beyond the port; item order of the samples follows the source file (the bootstrap depends on it).
"""
import collections
import math
import random

import numpy as np

from p2parts import debx as IO
from p2parts import answer_id as AID

HARD = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]
AGENTS = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "qwen/qwen3.7-plus", "claude:haiku", "claude:sonnet55"]


def majority(answers):
    if AID.RULE == "v8":
        c = collections.Counter(a for a in answers if a not in (None, ""))
        if not c: return None, 0
        return c.most_common(1)[0]
    top, cnt, _, _ = AID.vote(answers)
    return (top, cnt) if top is not None else (None, 0)


def maj_correct(rs):
    if AID.RULE == "v8":
        top, _ = majority([r.get("answer") for r in rs])
        if top is None: return False
        return bool(next((r for r in rs if r.get("answer") == top), {}).get("correct"))
    top, _, ok, _ = AID.vote([r.get("answer") for r in rs], [r.get("correct") for r in rs])
    return bool(ok) if top is not None else False


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def _mixed():
    items = collections.defaultdict(dict)
    for r in IO.ct_mixed(): items[r["id"]][(r["agent"], r["round"])] = r
    return items


def h3(items):
    D = {i: d for i, d in items.items() if all((a, r) in d for a in AGENTS for r in range(4))}
    cc = lambda d, r: sum(bool(d[(a, r)].get("correct")) for a in AGENTS)
    mo = lambda d, r: maj_correct([d[(a, r)] for a in AGENTS])
    out = dict(n=len(D), layers=dict(collections.Counter(next(iter(d.values()))["layer"] for d in D.values())), by_c={}, crossover=[])
    for c in range(7):
        its = [d for d in D.values() if cc(d, 0) == c]
        if not its: out["by_c"][str(c)] = dict(n=0); continue
        m = [sum(mo(d, r) for d in its) / len(its) for r in range(4)]
        lo, hi = wilson(sum(mo(d, 3) for d in its), len(its))
        if m[3] < m[0] and hi < m[0]: out["crossover"].append(c)
        out["by_c"][str(c)] = dict(n=len(its), maj=m, r3_wilson=[lo, hi], r3_minus_r0=m[3] - m[0])
    out["per_layer"] = {}
    for layer in HARD:
        cells = {}
        for c in range(7):
            its = [d for d in D.values() if next(iter(d.values()))["layer"] == layer and cc(d, 0) == c]
            if its: cells[str(c)] = dict(n=len(its), r0=sum(mo(d, 0) for d in its) / len(its), r3=sum(mo(d, 3) for d in its) / len(its))
        out["per_layer"][layer] = cells
    flips = {}
    for t in range(3):
        for m in range(6):
            cases = fl = 0
            for d in D.values():
                for a in AGENTS:
                    r = d[(a, t)]
                    if not r.get("correct"): continue
                    oth = [b for b in AGENTS if b != a]
                    k = AID.keys([r.get("answer")] + [d[(b, t)].get("answer") for b in oth])
                    if AID.RULE == "v8": k = [r.get("answer")] + [d[(b, t)].get("answer") for b in oth]   # raw, None included
                    if sum(1 for x in k[1:] if x != k[0]) != m: continue
                    cases += 1; fl += not d[(a, t + 1)].get("correct")
            flips[f"{t}->{t + 1}.m{m}"] = dict(cases=cases, share=fl / cases if cases else None)
    out["flips"] = flips
    out["element_error_r0"] = float(1 - np.mean([cc(d, 0) / 6 for d in D.values()]))
    out["share_ge4_r0"] = float(np.mean([cc(d, 0) >= 4 for d in D.values()]))
    out["n_ge4_r0"] = sum(cc(d, 0) >= 4 for d in D.values())
    # items with 1 or 2 of 6 correct at round 0: range of the round-3 majority rate; c >= 4 never lost (r3 rate 1.0)
    out["c12_r3_min"] = min(out["by_c"][c]["maj"][3] for c in ("1", "2"))
    out["c12_r3_max"] = max(out["by_c"][c]["maj"][3] for c in ("1", "2"))
    out["c_ge4_r3_min"] = min(out["by_c"][c]["maj"][3] for c in ("4", "5", "6") if out["by_c"][c]["n"])
    return out


def h6c(items):
    ds = collections.defaultdict(dict)                     # layer -> id -> {sample: row}, file order
    for r in IO.ct_deepseek_samples(): ds[r["layer"]].setdefault(r["id"], {})[r["sample"]] = r
    ds = {layer: ds[layer] for layer in HARD if layer in ds}
    homo = {arm: collections.defaultdict(dict) for arm in ("plain", "selfonly", "pressure")}
    for r in IO.ct_homo(): homo[r["arm"]][r["id"]][(r["agent"], r["round"])] = r

    def map_ok(v):
        if AID.RULE == "v8":
            top, _ = majority([v[s].get("answer") for s in range(16)])
            return bool(next(v[s] for s in range(16) if v[s].get("answer") == top).get("correct"))
        return maj_correct([v[s] for s in range(16)])
    rnd = random.Random(0)
    out = {}
    for layer in HARD + ["pooled"]:
        recs = []
        for ly, full in ds.items():
            if layer != "pooled" and ly != layer: continue
            for i, v in full.items():
                row = {"map": map_ok(v), "single": bool(v[0]["correct"])}
                for arm, it in homo.items():
                    d = it.get(i, {}); ags = sorted({a for (a, r) in d if r == 3})
                    row[arm] = maj_correct([d[(a, 3)] for a in ags]) if len(ags) >= 5 else None
                d = items.get(i, {})
                row["mixed"] = maj_correct([d[(a, 3)] for a in AGENTS]) if all((a, 3) in d for a in AGENTS) else None
                recs.append(row)
        if not recs: continue
        cell = dict(n=len(recs), map=float(np.mean([r["map"] for r in recs])), single=float(np.mean([r["single"] for r in recs])))
        for key in list(homo) + ["mixed"]:
            rs2 = [r for r in recs if r[key] is not None]
            if len(rs2) < 10: cell[key] = None; continue
            bs = sorted(np.mean([(lambda r: r[key] - r["map"])(rs2[rnd.randrange(len(rs2))]) for _ in rs2]) for _ in range(1000))
            cell[key] = dict(n=len(rs2), maj=float(np.mean([r[key] for r in rs2])),
                             delta=float(np.mean([r[key] for r in rs2]) - np.mean([r["map"] for r in rs2])), ci=[float(bs[25]), float(bs[974])])
        out[layer] = cell
    p = out["pooled"]
    out["homo_delta_max_pooled"] = max(p[a]["delta"] for a in ("plain", "selfonly", "pressure"))
    out["homo_delta_min_pooled"] = min(p[a]["delta"] for a in ("plain", "selfonly", "pressure"))
    return out


def run():
    items = _mixed()
    h = h3(items)
    pct = dict(c12_r3_min=100 * h["c12_r3_min"], c12_r3_max=100 * h["c12_r3_max"], share_ge4_r0=100 * h["share_ge4_r0"])
    return {"debate_bounds": dict(h3=h, h6c=h6c(items), pct=pct)}


if __name__ == "__main__":
    import json
    from pathlib import Path
    r = run()
    for k, v in r.items(): (Path(__file__).resolve().parents[1] / "summaries" / "out2" / f"{k}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False))
    h = r["debate_bounds"]
    print(h["h3"]["n"], h["h3"]["layers"], {c: (v["n"], v.get("maj")) for c, v in h["h3"]["by_c"].items()}, h["h3"]["share_ge4_r0"])
    print(json.dumps(h["h6c"]["pooled"], indent=1), h["h6c"]["homo_delta_max_pooled"])
