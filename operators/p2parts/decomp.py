"""Paths and loaders for the "decomp" area (v8 §9, decomposition). Reads only datasets/ (offline, $0).

datasets/set_a/decomposition/  (built by a maintainer build script (not published))
  runs.jsonl          560 decomposition attempts (main run, protocol v2): base, arm (arm2 = split, arm3 = team), rep, id, dataset,
                      correct, fail (plan failure), cost (USD, OpenRouter ledger), tokens (completion tokens over all calls),
                      n_subtasks, answered (a final answer was parsed), answer/gold (non-GPQA only), calls [{step, finish, served, out, worker, substituted}]
  pilot.jsonl         186 pilot attempts under protocol v1 (same fields; not pooled into any table)
  selection.json      97 rows [base, layer, id, stratum, correct-of-16, truncated-of-16]
  single-draws.csv    corpus draws of the two bases on the three layers (60 items x 16 draws each): correct (audited relabels
                      applied as in decomposition/stats.py), finish, out/think tokens, cost, answer (non-GPQA only;
                      empty for GPQA)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "datasets" / "set_a" / "decomposition"
RUNS = DIR / "runs.jsonl"
RUNS_B65536 = DIR / "runs.b65536.jsonl"   # 65,536-token equal-budget rerun (data not published); used only with P2_DECOMP_BUDGET=65536
PILOT = DIR / "pilot.jsonl"
SELECTION = DIR / "selection.json"
DRAWS = DIR / "single-draws.csv"
BASES = ("z-ai_glm-5.3-flash", "deepseek_deepseek-v3.2")


def _jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def runs(budget=8192):
    """budget 8192 = the preregistered run (runs.jsonl); 65536 = the x8 rerun (runs.b65536.jsonl)."""
    return _jsonl(RUNS_B65536 if budget == 65536 else RUNS)


def pilot():
    return _jsonl(PILOT)


def selection():
    """list of [base, layer, id, stratum, correct16, trunc16] in file order (the stats.py `meta` order)."""
    return json.load(open(SELECTION))


def draws(source="capped"):
    """list of dicts, file order (draw = 0-based order of the item's draws in the corpus file).
    source="capped" (default; §9 protocol) reads single-draws.csv whatever P2_DATA says; "uncapped" follows P2_DATA."""
    out = []
    from p2parts import datasrc
    path = datasrc.pick(DRAWS) if source == "uncapped" else DRAWS
    for r in csv.DictReader(open(path, encoding="utf-8")):
        r["draw"] = int(r["draw"])
        for k in ("correct_raw", "correct", "relabeled"):
            r[k] = int(r[k])
        r["out_tokens"] = int(r["out_tokens"]) if r["out_tokens"] else None
        r["think_tokens"] = int(r["think_tokens"]) if r["think_tokens"] else None
        r["cost"] = float(r["cost"]) if r["cost"] else None
        out.append(r)
    return out
