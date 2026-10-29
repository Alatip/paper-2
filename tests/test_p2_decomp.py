"""Regression tests for the "decomp" area (v8 §9).

  python3 -m pytest -q tests/test_p2_decomp.py     regression: operators/decomposition.run() reproduces STATS.md byte for byte
                                                  (the stored output of a private script (not published),
                                                  frozen in tests/expected2/decomp/), plus untraced numbers recomputed
                                                  in the 2026-10-10 review (maintainer workspace, not published).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "operators"))
sys.path.insert(0, str(ROOT / "summaries"))
import decomposition as DEC  # noqa: E402

EXP = ROOT / "tests" / "expected2" / "decomp"
_RES = {}


def res():
    if not _RES:
        _RES.update(DEC.run())
    return _RES


def test_stats_md_identical():
    assert res()["decomposition_stats.md"] == (EXP / "STATS.md").read_text()


def test_review_recomputed_numbers():
    """numbers recomputed by an independent review of the decomposition note (2026-10-10): truncation, protocol, cost."""
    u = res()["decomposition"]["untraced"]
    me = u["medium_easy"]
    assert (me["deepseek"]["arm2_sub_truncated"], me["deepseek"]["arm2_sub_calls"]) == (149, 242)
    assert (me["glm"]["arm2_sub_truncated"], me["glm"]["arm2_sub_calls"]) == (145, 648)
    assert round(me["deepseek"]["both_arms_sub_trunc_share"], 2) == 0.45 and round(me["glm"]["both_arms_sub_trunc_share"], 2) == 0.30
    assert me["deepseek"]["arm2_plan_fail"] == 35 and me["deepseek"]["arm2_plan_fail_all_plan_calls_length"] == 34
    assert me["deepseek"]["arm2_asm_truncated"] == 36 and me["deepseek"]["arm2_asm_truncated_no_answer"] == 34
    p = u["protocol"]
    assert (p["pilot_runs"], round(p["pilot_cost"], 3), round(p["main_cost"], 4)) == (186, 0.657, 5.0685)
    assert (p["max_attempt_tokens"], p["attempts_over_8192"], p["ds_hard_items_with_one_trunc"]) == (8207, 15, 7)
    h = u["ds_hard_true"]
    assert [round(h[k], 2) for k in ("single_p1_numeric", "single_p4_numeric", "single_p16_numeric", "arm2_p1_numeric", "arm2_p4_numeric",
                                     "arm3_p4_numeric")] == [0.06, 0.21, 0.50, 0.07, 0.21, 0.07]
    assert (h["zero16_gpqa_correct"], h["zero16_gpqa_attempts"]) == (9, 40)
    g = u["glm_cap"]
    assert (g["zero16_gpqa_correct"], g["zero16_gpqa_attempts"]) == (30, 64)
    d = u["dollars"]
    assert [round(d["deepseek"]["hard-true"][k], 2) for k in ("k1_raw", "k4_raw")] == [3.55, 14.19]
    assert [round(d["glm"]["hard-cap"][k], 2) for k in ("k1_raw", "k4_raw")] == [0.51, 2.03]
    assert round(d["deepseek"]["hard-true-numeric"]["k4_raw"], 2) == 9.97
    assert [round(d["deepseek"]["hard-true"][k], 2) for k in ("single_at_k1", "single_at_k4")] == [0.17, 0.38]
    assert round(d["deepseek"]["hard-true-numeric"]["single_at_k4"], 2) == 0.39 and round(d["glm"]["hard-cap"]["single_at_k4"], 2) == 0.18


def test_results_md_dollar_table():
    """RESULTS.md «Dollars» first table: $ / tokens / ratios at the printed rounding."""
    d = res()["decomposition"]["untraced"]["dollars"]
    want = {("deepseek", "hard-true"): (0.0046, 3255, 0.0163, 4660, 0.0132, 4427, 3.5, 2.9),
            ("deepseek", "medium"): (0.0032, 2632, 0.0247, 5899, 0.0163, 5662, 7.7, 5.1),
            ("deepseek", "easy"): (0.0009, 900, 0.0086, 2957, 0.0092, 3842, 9.1, 9.7),
            ("glm", "hard-cap"): (0.0042, 7735, 0.0021, 2987, 0.0102, 4763, 0.5, 2.4),
            ("glm", "medium"): (0.0024, 4699, 0.0010, 1386, 0.0105, 4499, 0.4, 4.4),
            ("glm", "easy"): (0.0004, 767, 0.0007, 836, 0.0073, 3183, 1.6, 17.4)}
    for (b, s), w in want.items():
        r = d[b][s]
        got = (round(r["draw_cost"], 4), round(r["draw_tokens"]), round(r["arm2_cost"], 4), round(r["arm2_tokens"]),
               round(r["arm3_cost"], 4), round(r["arm3_tokens"]), round(r["arm2_ratio"], 1), round(r["arm3_ratio"], 1))
        assert got == w, (b, s, got, w)
