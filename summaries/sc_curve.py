"""Self-consistency curve: plurality accuracy of k draws of one model, k = 1..16, on the uncapped 16-draw data
(datasets/set_a/tn5-samples/rows.uncapped.jsonl.gz), paper-1 answer rule (operators/p2parts/answer_id). For each item with all 16
draws, average over R random k-subsets; a tie at the top is credited 1/m (paper-1 rule 'split'); also the tie rate (top
answer not unique) and the "no majority" rate (top count < k/2). Purpose: choose k for paper 2 (2026-10-20). $0.
Usage: python3 summaries/sc_curve.py   -> summaries/out2/sc_curve.md"""
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "operators"))
from p2parts import answer_id as AID  # noqa: E402

KS = (1, 2, 3, 4, 5, 6, 7, 9, 12, 16)
R = 300


def load():
    by = defaultdict(dict)                 # (model, layer, id) -> {sample: (answer, ok)}
    with gzip.open(ROOT / "datasets" / "set_a" / "tn5-samples" / "rows.uncapped.jsonl.gz", "rt") as h:
        for r in map(json.loads, filter(str.strip, h)):
            by[(r["model"], r["layer"], r["id"])][r["sample"]] = (r.get("answer"), bool(r.get("correct")))
    return by


def item_curve(draws, rng):
    ans = [a for a, _ in draws]
    oks = [o for _, o in draws]
    keys = AID.keys(ans)                   # identity groups over all 16 (chaining within the item)
    right = AID.right_keys(keys, oks)
    out = {}
    for k in KS:
        acc = tie = nomaj = 0.0
        for _ in range(R if k < len(draws) else 1):
            idx = rng.sample(range(len(draws)), k)
            c = Counter(keys[i] for i in idx if keys[i] is not None)
            if not c:
                continue
            top = max(c.values())
            tops = [x for x, v in c.items() if v == top]
            acc += sum(x in right for x in tops) / len(tops)
            tie += len(tops) > 1
            nomaj += top * 2 <= k
        n = R if k < len(draws) else 1
        out[k] = (acc / n, tie / n, nomaj / n)
    return out


def main():
    rng = random.Random(0)
    by = load()
    agg = defaultdict(lambda: defaultdict(list))   # (model, layer) -> k -> [(acc, tie, nomaj)]
    for (m, L, i), d in by.items():
        if len(d) < 16:
            continue
        cur = item_curve([d[s] for s in sorted(d)][:16], rng)
        for k, v in cur.items():
            agg[(m, L)][k].append(v)
            agg[(m, "ALL")][k].append(v)
    rows = ["| model | set | N | " + " | ".join(f"k={k}" for k in KS) + " | tie@3 | tie@5 | nomaj@3 | nomaj@5 |",
            "|---|---|---|" + "---|" * (len(KS) + 4)]
    for (m, L) in sorted(agg):
        A = agg[(m, L)]
        n = len(A[1])
        mean = lambda k, j: sum(v[j] for v in A[k]) / n
        rows.append(f"| {m} | {L} | {n} | " + " | ".join(f"{mean(k, 0):.3f}" for k in KS) +
                    f" | {mean(3, 1):.2f} | {mean(5, 1):.2f} | {mean(3, 2):.2f} | {mean(5, 2):.2f} |")
    txt = "\n".join(rows) + "\n"
    (ROOT / "summaries" / "out2").mkdir(exist_ok=True)
    (ROOT / "summaries" / "out2" / "sc_curve.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
