"""Regression tests for the "budget" area (v8 §10, cost-matched Q1–Q5).

The operator must reproduce the stored results.json of a private script (not published) (tests/expected2/budget/) to 1e-9,
including bootstrap intervals (same seed and call order), and the RESULTS.md tables at their printed rounding.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "operators"))
sys.path.insert(0, str(ROOT / "summaries"))
EXP = ROOT / "tests" / "expected2" / "budget"
_CACHE = {}


def out():
    """Under the OLD flags (P2_ANSWER_RULE=v8: run.py's labels; P2_MATRIX=tn5: stored capped T=0 row), which produced results.json."""
    if "b" not in _CACHE:
        import budget
        from p2parts import answer_id as AID
        prev = (AID.RULE, budget.MATRIX_SOURCE)
        AID.set_rule("v8"); budget.MATRIX_SOURCE = "tn5"
        try:
            _CACHE["b"] = budget.run()["budget"]
        finally:
            AID.set_rule(prev[0]); budget.MATRIX_SOURCE = prev[1]
    return _CACHE["b"]


def test_default_flags_paper1_rule_and_corpus():
    """Default = paper 1's answer rule over the stored akey + uncapped T=0 answer; Q1/Q2/Q5 do not depend on either flag."""
    import budget
    from p2parts import answer_id as AID
    prev = (AID.RULE, budget.MATRIX_SOURCE)
    AID.set_rule("p1"); budget.MATRIX_SOURCE = "corpus"
    try:
        new = budget.run()["budget"]
    finally:
        AID.set_rule(prev[0]); budget.MATRIX_SOURCE = prev[1]
    old = out()
    for q in ("q1", "q2", "q5"):
        bad = []; _cmp(new[q], old[q], q, bad)
        assert not bad, bad[:5]
    assert sum(new["q3"][L]["single_from_corpus"] for L in new["q3"]) > 0
    for L in new["q3"]:   # same item sets
        assert new["q3"][L]["n_items"] == old["q3"][L]["n_items"]


def _cmp(x, y, path, bad):
    if isinstance(y, dict):
        for k in y:
            if k not in x: bad.append((path + "." + k, "missing"))
            else: _cmp(x[k], y[k], path + "." + k, bad)
    elif isinstance(y, list):
        if len(x) != len(y): bad.append((path, "len"))
        for i, (u, v) in enumerate(zip(x, y)): _cmp(u, v, f"{path}.{i}", bad)
    elif y is None:
        if x is not None: bad.append((path, x, y))
    elif abs(float(x) - float(y)) > 1e-9:
        bad.append((path, x, y))


def test_results_json_bit_for_bit():
    stored = json.load(open(EXP / "results.json"))
    bad = []
    for q in ("q1", "q2", "q3", "q4", "q5"): _cmp(out()[q], stored[q], q, bad)
    assert not bad, bad[:10]


def test_results_md_q1_rows():
    """Q1 table rows of RESULTS.md (printed at 3 decimals / 2 for CIs)."""
    md = (EXP / "RESULTS.md").read_text()
    nm = {"MMLU-Pro": "mmlu-pro", "GPQA": "gpqa-diamond", "GSM-Hard": "gsm-hard", "olympiad": "olympiad"}
    n = 0
    for m in re.finditer(r"^\| (MMLU-Pro|GPQA|GSM-Hard|olympiad) \| (\d+) \| [^|]+\| \d+ \| (\d+) \| ([\d.]+) \[[^|]+\| ([\d.]+) \[", md, re.M):
        L, B = nm[m.group(1)], m.group(3)
        d = out()["q1"][L]["B"][B]
        assert int(m.group(2)) == out()["q1"][L]["n"]
        assert f"{d['lower'][0]:.3f}" == m.group(4) and f"{d['upper'][0]:.3f}" == m.group(5)
        n += 1
    assert n == 24


def test_section10_ranges():
    s = out()["s10"]
    assert s["10_3"]["n_cells"] == 120
    assert f"{s['10_3']['worst_gain_pts']:.0f}" == "-68"
    assert s["10_4"]["olymp_single_stopped"] == 89
