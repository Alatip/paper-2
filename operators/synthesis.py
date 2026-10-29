"""Synthesis / aggregation: one DeepSeek V3.2 call reads several answers and writes one final answer (paper v8 §6.3 last
paragraph "Stopping and synthesis", restated in §6.7; table B4/C4 of tech-note-5 results/tables.md).

Two modes (port of tech-note-5 stats.section_aggregate and the synthesiser line of extra.section_fifth):
- synth: inputs = the six mixed-debate round-3 answers;
- moa:   inputs = five of DeepSeek's own T=1 draws (v8 calls this "synthesis in one DeepSeek call" and quotes its
         "0 of 16": items on which no input draw was correct).
Per mode, on rows with a graded output (correct is not None): input majority correct (plurality by answer identity of the
graded inputs, correct if any input carrying the top answer is correct), aggregator accuracy, any-input-correct rate,
aggregator correct when no input was correct (k / n), novel answers (not among inputs; flag stored by the run) and
novel & correct. GPQA answers are per-item letters / opaque codes (identity preserved).
Data: datasets/set_a/tn5-aggregate. Offline, $0.
"""
from p2parts import tn5 as IO


def table(mode, layers=IO.HARD):
    rs = [r for L in layers for r in IO.aggregate(mode, L) if r.get("correct") is not None]
    if not rs: return None
    mv = 0
    for r in rs:
        ins = [x for x in r["inputs"] if x.get("correct") is not None]
        if IO.AID.RULE == "v8":
            top, _ = IO.majority([x["answer"] for x in ins])
            mv += any(IO.canon(x["answer"]) == top and x["correct"] for x in r["inputs"])
        else:
            mv += IO.maj_correct(ins)
    none = [r for r in rs if not r["any_input_correct"]]; nov = [r for r in rs if r["novel"]]
    return {"items": len(rs), "input_majority": mv / len(rs), "aggregator": sum(bool(r["correct"]) for r in rs) / len(rs),
            "any_input": sum(r["any_input_correct"] for r in rs) / len(rs), "none_correct_items": len(none),
            "none_correct_recovered": sum(bool(r["correct"]) for r in none), "novel": len(nov), "novel_correct": sum(bool(r["correct"]) for r in nov),
            "none_correct_ids": sorted(r["id"] for r in none)}


def run() -> dict:
    return {"synthesis": {m: table(m) for m in ("synth", "moa")}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1))
