"""Paths and loaders for the "budget" area (paper 2, v8 §10). Reads only datasets/budget-*/ (relative to the repository root).

Answer keys ("akey", added 2026-10-18 for paper 1's answer rule): letters (MMLU-Pro, GPQA) or opaque ~k codes for an answer with no
letter; the parsed answer string elsewhere; None = no answer. "ok" = run.py's grader marks that answer correct.
Answer labels ("vote"): GOLD = graded correct, W1, W2, ... = distinct wrong answers of that item, None = no parsable answer.
Rows are returned in source-file order (the original script's bootstrap depends on it).
"""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "datasets"
FORCING = D / "set_a" / "budget-forcing" / "rows.csv"
CUT = D / "set_a" / "budget-deepseek-cut" / "cut-correct.jsonl.gz"
NATURAL = D / "set_a" / "budget-deepseek-cut" / "natural.csv"
DEBATES = D / "set_a" / "budget-debates" / "rows.csv"
CLAUDE = D / "set_a" / "budget-claude-cost" / "rows.csv"
LAYERS = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]


def _csv(p):
    from p2parts import datasrc
    with open(datasrc.pick(p), newline="") as fh:   # P2_DATA: uncapped twin (default) or capped original
        return list(csv.DictReader(fh))


def _i(x): return int(x) if x not in ("", None) else None
def _f(x): return float(x) if x not in ("", None) else None
def _v(x): return x or None


def forcing():
    return [dict(layer=r["layer"], id=r["id"], budget=int(r["budget"]), sample=_i(r["sample"]), correct=r["correct"] == "1",
                 finish=r["finish"] or None, out_tok=_i(r["out_tok"])) for r in _csv(FORCING)]


def cut_rows():
    """DeepSeek native traces (T=0 'single', T=1 'samples') with lower/upper correctness and vote label after a cut at each B."""
    with gzip.open(CUT, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def natural():
    return [dict(layer=r["layer"], stage=r["stage"], id=r["id"], sample=_i(r["sample"]), correct=r["correct"] == "1",
                 finish=r["finish"] or None, out_tok=_i(r["out_tok"]), cost=_f(r["cost_usd"]), vote=_v(r["vote"]),
                 akey=_v(r.get("akey")), ok=r["correct"] == "1") for r in _csv(NATURAL)]


def debates():
    return [dict(layer=r["layer"], kind=r["kind"], id=r["id"], round=int(r["round"]), agent=r["agent"], finish=r["finish"] or None,
                 out_tok=_i(r["out_tok"]), vote=_v(r["vote"]),
                 akey=_v(r.get("akey")), ok=r.get("ok") == "1") for r in _csv(DEBATES)]


def claude():
    return [dict(model=r["model"], layer=r["layer"], stage=r["stage"], id=r["id"], sample=_i(r["sample"]), correct=r["correct"] == "1",
                 cost=_f(r["cost_usd"]), out_tok=_i(r["out_tok"]), think_tok=_i(r["think_tok"]), vote=_v(r["vote"]),
                 akey=_v(r.get("akey")), ok=r["correct"] == "1") for r in _csv(CLAUDE)]


def corpus_deepseek(layer):
    """{id: row} of DeepSeek V3.2 in paper 1's uncapped T=0 matrix (datasets/set_a/corpus-<layer>/answers.csv, label audits applied;
    a 65,536-cap hit is graded wrong). Used for Q3's natural T=0 answer (P2_MATRIX=corpus, default)."""
    import sys
    sys.path.insert(0, str(ROOT / "lib"))
    from p1stats import data as _D
    return {i: d["deepseek_deepseek-v3.2"] for i, d in _D.load_layer(layer).items()}


# ---------- dollar mode (decided 2026-10-19; datasets/model-prices) ----------
PRICES = D / "model-prices" / "prices.csv"
MATRIX = D / "set_a" / "tn5-matrix" / "rows.jsonl.gz"
SAMPLES = D / "set_a" / "tn5-samples" / "rows.jsonl.gz"
HOMO = D / "set_a" / "tn5-debate-homo" / "rows.jsonl.gz"
MIXED = D / "set_a" / "tn5-debate-mixed" / "rows.jsonl.gz"
SERVED = {"deepseek/deepseek-v3.2": "deepseek_deepseek-v3.2", "z-ai/glm-5.3-flash": "z-ai_glm-5.3-flash",
          "qwen/qwen3.7-plus": "qwen_qwen3.7-plus", "mistralai/mistral-large-2512": "mistralai_mistral-large-2512",
          "claude:haiku": "claude_haiku", "claude:sonnet55": "claude_sonnet55", "claude:opus": "claude_opus"}


def _jl(p):
    from p2parts import datasrc
    with gzip.open(datasrc.pick(p), "rt") as fh:   # P2_DATA: uncapped twin (default) or capped original
        return [json.loads(l) for l in fh if l.strip()]


def prices():
    """{model: (pi_in, pi_out)} in $ per token: the provider=ALL rows of datasets/model-prices/prices.csv."""
    with open(PRICES, newline="") as fh:
        return {r["model"]: (float(r["pi_in_per_M"]) / 1e6, float(r["pi_out_per_M"]) / 1e6) for r in csv.DictReader(fh)
                if r["provider"] == "ALL"}


def matrix_tokens():
    """{(id, model): (tok_in, tok_out)} of the T=0 matrix (round 0 of the mixed debate; same prompt as paper 1's corpus answer)."""
    return {(r["id"], r["model"]): (r["tok_in"], r["tok_out"]) for r in _jl(MATRIX)}


def sample_tokens(model="deepseek_deepseek-v3.2"):
    """{(id, sample): (tok_in, tok_out)} of the T=1 draws of `model` (tn5-samples)."""
    return {(r["id"], r["sample"]): (r["tok_in"], r["tok_out"]) for r in _jl(SAMPLES) if r["model"] == model}


def debate_tokens(kind):
    """{(id, round, agent): (model, tok_in, tok_out)}, rounds 1-3: 'copies' = homo debate arm plain, 'families' = mixed debate."""
    if kind == "copies":
        return {(r["id"], r["round"], r["agent"]): ("deepseek_deepseek-v3.2", r["tok_in"], r["tok_out"])
                for r in _jl(HOMO) if r["arm"] == "plain"}
    return {(r["id"], r["round"], r["agent"]): (SERVED[r["agent"]], r["tok_in"], r["tok_out"]) for r in _jl(MIXED)}


def pool_layer(layer):
    """{id: {model: row}} of paper 1's uncapped 7-model corpus (label audits applied): the natural single answers for best(B)."""
    import sys
    sys.path.insert(0, str(ROOT / "lib"))
    from p1stats import data as _D
    return _D.load_layer(layer)
