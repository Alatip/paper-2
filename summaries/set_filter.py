"""Which item sets of the paper-1 corpus pass the stand filter (2026-10-19):
  (1) the weakest model of the pool errs noticeably (its accuracy <= WEAK_MAX), and
  (2) the prize in the pool is not small (oracle - best >= PRIZE_MIN).
Per layer, per category / source within a layer (MMLU-Pro categories, GPQA domains, AIME vs HMMT), per pool P7/P4/P3.
Uncapped T=0 answers of the 7 models (paper 1). $0. Writes summaries/out2/set_filter.json/.md.
Usage: python3 summaries/set_filter.py [WEAK_MAX PRIZE_MIN]"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
from p1stats import data as D  # noqa: E402

NO_MISTRAL = "--no-mistral" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
WEAK_MAX, PRIZE_MIN = (float(_args[0]), float(_args[1])) if len(_args) > 1 else (0.80, 0.05)
MISTRAL = "mistralai_mistral-large-2512"


def pools():
    """P7/P4/P3 of paper 1; with --no-mistral: P6 = P7 - Mistral, W3 = Haiku+DeepSeek+GLM, W2 = Haiku+DeepSeek."""
    if not NO_MISTRAL:
        return {P: D.POOLS[P] for P in ("P7", "P4", "P3")}
    return {"P6": [m for m in D.POOLS["P7"] if m != MISTRAL], "W3": [m for m in D.POOLS["P4"] if m != MISTRAL],
            "W2": [m for m in D.POOLS["P3"] if m != MISTRAL]}
OUT = ROOT / "summaries" / "out2"


def groups():
    g = {}
    for L in D.LAYERS:
        for r in csv.DictReader(open(ROOT / "datasets" / "set_a" / f"corpus-{L}" / "items.csv")):
            cat = r.get("category") or ""
            if L == "olympiad":
                cat = r["id"].split(":")[0]
            g[r["id"]] = (L, cat)
    return g


def stats(X, ids, pool):
    C = np.array([[float(bool(X[i][m]["correct"])) for i in ids] for m in pool])
    acc = C.mean(1)
    best, oracle = acc.max(), C.max(0).mean()
    return dict(N=len(ids), weakest=float(acc.min()), weakest_model=pool[int(acc.argmin())], best=float(best),
                best_model=pool[int(acc.argmax())], oracle=float(oracle), prize=float(oracle - best),
                passes=bool(acc.min() <= WEAK_MAX and oracle - best >= PRIZE_MIN))


def run():
    A, G, R = {L: D.load_layer(L) for L in D.LAYERS}, groups(), {}
    for L, X in A.items():
        cats = defaultdict(list)
        for i in X:
            cats[G.get(i, (L, "?"))[1]].append(i)
        for P, pool in pools().items():
            R[f"{L} | all | {P}"] = stats(X, list(X), pool)
            for c, ids in sorted(cats.items()):
                if len(ids) >= 10 and len(cats) > 1:
                    R[f"{L} | {c} | {P}"] = stats(X, ids, pool)
    return R


def md(R):
    rows = [f"Filter: weakest <= {WEAK_MAX:.2f} and prize >= {100 * PRIZE_MIN:.0f} pp.", "",
            "| set | group | pool | N | weakest (model) | best (model) | oracle | prize, pp | passes |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in R.items():
        L, c, P = k.split(" | ")
        rows.append(f"| {L} | {c} | {P} | {s['N']} | {s['weakest']:.3f} ({s['weakest_model']}) | {s['best']:.3f} ({s['best_model']}) | "
                    f"{s['oracle']:.3f} | {100 * s['prize']:.1f} | {'yes' if s['passes'] else ''} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    R = run()
    OUT.mkdir(exist_ok=True)
    tag = "_no_mistral" if NO_MISTRAL else ""
    json.dump(R, open(OUT / f"set_filter{tag}.json", "w"), indent=1)
    (OUT / f"set_filter{tag}.md").write_text(md(R))
    print(md(R))
