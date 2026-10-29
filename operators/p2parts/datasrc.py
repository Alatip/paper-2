"""Which version of a rerun dataset file the paper-2 loaders read (decided 2026-10-18).

P2_DATA = "uncapped" (default): where datasets/<name>/ holds a twin `<stem>.uncapped.<ext>`, read it. The twin is the original
  file with the rows that hit the 8,192-token cap replaced by the uncapped rerun (max 65,536 output tokens, same prompts; debates
  rerun from the first cut round on for all agents of the item; round-0 T=0 answers from paper 1's uncapped matrix). Each
  replaced row carries `src` (rerun2 / rerun2-length / p1rerun / p1rerun-length; *-length = hit 65,536 again, graded wrong).
P2_DATA = "capped": the original files (the v8 data); the pinned v8 regressions run under this flag.
Env read at import; set_data() switches at run time (loaders that cache must be cleared by the caller).
"""
import os
from pathlib import Path

DATA = os.environ.get("P2_DATA", "uncapped")
assert DATA in ("uncapped", "capped"), DATA


def set_data(v):
    global DATA
    assert v in ("uncapped", "capped"), v
    DATA = v


def twin(path):
    p = Path(path)
    name = p.name
    stem, dot, ext = name.partition(".")
    return p.with_name(f"{stem}.uncapped.{ext}")


def pick(path):
    """The file to read for `path` under the current flag."""
    if DATA == "uncapped":
        t = twin(path)
        if t.exists(): return t
    return Path(path)
