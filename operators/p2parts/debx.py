"""Area "debx" (v8 §6.4–§6.7): path constants and loaders. Reads only datasets/debx-*/ (built by a maintainer build script (not published)).

Rows are returned as plain dicts with the source field names (id, layer, agent, round, answer, correct, ...), in file order, so the
"last row per key wins" rule of the original loaders is kept by the callers.
"""
import gzip
import json
from pathlib import Path

from p2parts import datasrc

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
WEAK = DATASETS / "set_a" / "debx-weak-pool"
SIX = DATASETS / "set_a" / "debx-six-debate"
CTD = DATASETS / "set_a" / "debx-ct-debate"


def jsonl(path):
    path = datasrc.pick(path)   # P2_DATA: uncapped twin if present (default) or the capped original
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for l in f:
            if l.strip():
                try: yield json.loads(l)
                except json.JSONDecodeError: pass


# weak pool: round-0 matrix rows of W on the 733 items, item groups, the three new runs
def weak_items():          return list(jsonl(WEAK / "items.jsonl.gz"))
def weak_round0():         return list(jsonl(WEAK / "round0.jsonl.gz"))
def weak_debate():         return list(jsonl(WEAK / "debate-w.jsonl.gz"))
def weak_synth():          return list(jsonl(WEAK / "synth-w.jsonl.gz"))
def weak_judge():          return list(jsonl(WEAK / "judge-w.jsonl.gz"))   # judges: not reported in paper 2


# six-agent mixed debate as tech-note-5 stats.py sees it (label audit applied; 225 items complete in all rounds)
def six_debate():          return list(jsonl(SIX / "debate6.jsonl.gz"))


# corpus-traces live rows as axis-H-bounds sees them (no label audit, raw answer strings)
def ct_mixed():            return list(jsonl(CTD / "mixed.jsonl.gz"))
def ct_homo():             return list(jsonl(CTD / "homo.jsonl.gz"))
def ct_deepseek_samples(): return list(jsonl(CTD / "deepseek-samples16.jsonl.gz"))


