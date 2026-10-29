"""Run every set-A operator of paper 2 ($0, offline, data only from datasets/) and write summaries/out2/*.json|md."""
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "operators"))
sys.path.insert(0, str(ROOT / "summaries"))

OUT2 = ROOT / "summaries" / "out2"
# operator module -> v8 sections (order = paper order)
OPERATORS = [
    ("self_consistency", "§5.7"), ("diversity", "§5.8"), ("debate", "§6.3"), ("synthesis", "§6.3"),
    ("debate_weak", "§6.4"), ("debate_bounds", "§6.6"),
    ("decomposition", "§9"), ("budget", "§10"), ("core_report", "core"),
]


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def run_operators():
    OUT2.mkdir(parents=True, exist_ok=True)
    log = []
    for mod, sec in OPERATORS:
        t = time.time()
        m = importlib.import_module(mod)
        res = m.run()
        for stem, val in res.items():
            if stem.endswith(".md"):
                (OUT2 / stem).write_text(val if val.endswith("\n") else val + "\n")
            else:
                (OUT2 / f"{stem}.json").write_text(json.dumps(val, indent=1, ensure_ascii=False, sort_keys=True,
                                                              default=_default) + "\n")
        log.append(f"{mod} ({sec}): {', '.join(sorted(res))} [{time.time() - t:.1f}s]")
        print(log[-1], flush=True)
    return log


if __name__ == "__main__":
    run_operators()
