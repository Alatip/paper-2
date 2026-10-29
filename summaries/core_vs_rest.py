"""Representativeness check of the paper-2 core (first 60 items per layer, file order) against the rest of the paper-1
corpus, on the uncapped T=0 answers of the 7 models: per layer and pool — N, best (in-sample), oracle, prize, plurality
vote (ties to the first participant), share of prize, mean phi. $0, offline. Writes summaries/out2/core_vs_rest.json/.md.
Usage: python3 summaries/core_vs_rest.py"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
from p1stats import data as D, voting as V  # noqa: E402

OUT = ROOT / "summaries" / "out2"


def core_ids():
    with gzip.open(ROOT / "datasets" / "set_a" / "tn5-samples" / "rows.jsonl.gz", "rt") as h:
        return {(r["layer"], r["id"]) for r in map(json.loads, filter(str.strip, h))}


def stats(X, ids, pool):
    C = np.array([[float(bool(X[i][m]["correct"])) for i in ids] for m in pool])
    best, oracle = C.mean(1).max(), C.max(0).mean()
    vote = float(np.mean([V.plurality_rows(X[i], pool, "first") for i in ids]))   # plurality, ties to the first participant
    ph = [np.corrcoef(C[a], C[b])[0, 1] for a in range(len(pool)) for b in range(a + 1, len(pool))
          if C[a].std() > 0 and C[b].std() > 0]
    prize = oracle - best
    return dict(N=len(ids), best=best, oracle=oracle, prize=prize, vote=vote,
                share=(vote - best) / prize if prize > 0 else None, phi=float(np.mean(ph)) if ph else None)


def run():
    core, A, R = core_ids(), {L: D.load_layer(L) for L in D.LAYERS}, {}
    for L, X in A.items():
        ins = [i for i in X if (L, i) in core]
        out = [i for i in X if (L, i) not in core]
        for P in ("P7", "P4", "P3"):
            pool = D.POOLS[P]
            R[f"{L}.{P}"] = {"core": stats(X, ins, pool), "rest": stats(X, out, pool)}
    return R


def md(R):
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"
    rows = ["| set.pool | part | N | best | oracle | prize | vote | share | φ̄ |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in R.items():
        for part in ("core", "rest"):
            s = v[part]
            rows.append(f"| {k} | {part} | {s['N']} | {f(s['best'])} | {f(s['oracle'])} | {f(s['prize'])} | {f(s['vote'])} | "
                        f"{f(s['share'], 2)} | {f(s['phi'], 2)} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    R = run()
    OUT.mkdir(exist_ok=True)
    json.dump(R, open(OUT / "core_vs_rest.json", "w"), indent=1)
    (OUT / "core_vs_rest.md").write_text(md(R))
    print(md(R))
