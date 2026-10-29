"""Regression tests, area tn5: the operators self_consistency / diversity / debate / synthesis re-render the rows of the
tech-note-5 tables they port (A', A'2, A2, D, B, B2, B3, B5, B4/C4, T, N) in the original print format, and every rendered
line must occur verbatim in the frozen stored output tests/expected2/tn5/tables.md (TN5 run.py output)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "operators"))
import self_consistency, diversity, debate, synthesis  # noqa: E402

TABLES = (ROOT / "tests" / "expected2" / "tn5" / "tables.md").read_text()
MAIN_PART = TABLES.split("# L. Legacy tables")[0]


@pytest.fixture(scope="module", autouse=True)
def _old_answer_rule():
    """The frozen outputs were produced with the v8 answer identity: replay them under P2_ANSWER_RULE=v8 (the default is
    paper 1's rule, tested in test_p2_answer_rule.py)."""
    from p2parts import answer_id as AID
    from p2parts import tn5 as T
    prev = AID.RULE; AID.set_rule("v8")
    prevm = T.MATRIX_SOURCE; T.set_matrix("tn5")   # and on the capped tech-note-5 matrix (P2_MATRIX=tn5)
    yield
    AID.set_rule(prev); T.set_matrix(prevm)


def test_family_triples_default_corpus():
    """Default matrix = paper 1's uncapped corpus: every corpus item enters (553/164/141/83), 35 triples per layer."""
    from p2parts import tn5 as T
    prevm = T.MATRIX_SOURCE; T.set_matrix("corpus")
    try:
        f = diversity.family_triples()
        o = self_consistency.untraced_olympiad_deepseek()
    finally:
        T.set_matrix(prevm)
    assert {L: v["n"] for L, v in f.items()} == {"mmlu-pro": 553, "gpqa-diamond": 164, "gsm-hard": 141, "olympiad": 83}
    assert all(v["triples"] == 35 for v in f.values())
    assert o["items"] + len(o["items_dropped_not_in_corpus"]) == 49


@pytest.fixture(scope="module")
def out():
    d = {}
    for m in (self_consistency, diversity, debate, synthesis): d.update(m.run())
    return d


def has(line):
    assert line in MAIN_PART, f"not in tables.md: {line}"


def test_samples_a1(out):
    for v in out["self_consistency"]["A1"].values():
        for L, r in v.items():
            has(f"| {r['model']} | {L} | {r['items']} | {r['k']} | {r['p']:.3f} | {r['mv']:.3f} | {r['passk']:.3f} | {r['all_wrong']:.3f} | {r['phi']:.2f} | {r['n_eff']:.2f} | {r['all_n']} / {r['all_mv']:.3f} / {r['all_passk']:.3f} |")


def test_samples_a2(out):
    for v in out["self_consistency"]["A2"].values():
        for L, r in v.items():
            t = f"{r['tetra']:.2f}" if r["tetra"] is not None else "–"
            has(f"| {r['model']} | {L} | {r['items']} | {r['icc1']:.2f} | {t} | {r['mv1']:.3f} | {r['mv4']:.3f} | {r['mv8']:.3f} | {r['mvk']:.3f} | {r['kish']:.3f} | {r['pass4']:.3f} → {r['passk']:.3f} |")


def test_family_a2(out):
    for L, r in out["diversity"]["family"].items():
        has(f"| {L} | {r['n']} | {r['tetra_mean']:.2f} | {r['tetra_min']:.2f}–{r['tetra_max']:.2f} | {r['wrong_q95']} | {r['all_wrong']:.3f} | {r['all_wrong_indep']:.4f} | {r['gain_best']:+.3f} / {r['gain_worst']:+.3f} / {r['gain_mean']:+.3f} ({r['triples']} triples) | {r['spearman']:+.2f} |")


def test_personas_d(out):
    for L, r in out["diversity"]["personas"].items():
        if L.startswith("_"): continue
        has(f"| {L} | {r['items']} | {r['p_plain']:.3f} | {r['p_personas']:.3f} | {r['phi_within_plain']:.2f} | {r['phi_within_persona']:.2f} | {r['phi_across']:.2f} | {r['n_eff_personas']:.2f} | {r['mv_plain4']:.3f} | {r['mv_personas']:.3f} | {r['diff']:+.2f} [{r['diff_lo']:+.2f}, {r['diff_hi']:+.2f}] ({r['undefined']} undefined) |")


def test_debate_b(out):
    for n, title in (("6", "6 agents"), ("5", "5 agents (no Mistral)")):
        b = out["debate"]["mixed"]["B"][n]
        has(f"### {title}: {b['items']} items")
        for rd, r in b["rounds"].items():
            has(f"| {rd} | {r['agent_acc']:.3f} | {r['majority']:.3f} | {r['unanimous']:.3f} | {r['fixed']} | {r['broken']} | {r['truncated']} |")


def test_debate_b2(out):
    b2 = out["debate"]["mixed"]["B2"]
    for rd, r in b2["rounds"].items():
        has(f"| {rd} | {r['phi']:.2f} | {r['tetra']:.2f} ({r['tetra_pairs']}/15) | {r['icc1']:.2f} | {r['n_eff']:.2f} | {r['unanimous']} | {r['unanimous_wrong']} | {r['all_wrong']} | {r['items_6']} |")
    for k, r in b2["flips"].items():
        has(f"| {k} | {r['cases']} | {r['rate']:.2f} |")
    t = b2["tokens_per_turn"]
    has("Mean completion tokens per agent-turn: " + ", ".join(f"round {rd} {v:.0f}" for rd, v in t.items()) + f"; 3 rounds × 6 agents ≈ {b2['tokens_per_item_3_rounds']:.0f} tokens per item")


def test_debate_b3(out):
    b3 = out["debate"]["mixed"]["B3"]
    for L, r in b3["layers"].items():
        has(f"| {L} | {r['items']} | {r['phi0']:.2f} | {r['n_eff0']:.2f} | {r['maj0']:.3f} | {r['maj3']:.3f} | {r['best0_trunc0']:.3f} | {r['gain_vs_best0']:+.3f} | {r['best_answered']:.3f} ({r['best_answered_items']} items) | {r['gain_vs_best_answered']:+.3f} | {r['gain_vs_maj0']:+.3f} |")
    s = b3["stopping"]
    has(f"Accuracy {s['accuracy']:.3f} using {s['mean_rounds']:.2f} rounds on average (vs 3). Stops: " + ", ".join(f"round {rd}: {v['items']} items, {v['wrong']} wrong" for rd, v in s["stops"].items()) + ".")


def test_debate_t(out):
    t = out["debate"]["mixed"]["T"]
    for name, r in t["rules"].items():
        per = " | ".join(f"{v['maj0']:.3f} → {v['maj3']:.3f}" for v in r["layers"].values())
        has(f" | {r['maj0']:.3f} | {r['maj3']:.3f} | {r['gain']:+.3f} | {per} |")
    for L, r in t["best_answered"].items():
        has(f"| {L} | {r['agent']} | {r['answered']} of {r['items']} | {r['acc']:.3f} | {r['maj0']:.3f} | {r['maj3']:.3f} | {r['maj3_minus_agent']:+.3f} | {r['maj3_all']:.3f} |")


def test_debate_n(out):
    n = out["debate"]["mixed"]["N"]
    has(f"- Mixed debate, {n['items']} items: {n['fixes']} fixes, {n['fixes_cut']} of them on items with at least one cut-off round-0 row. On the {n['six_items']} items all six answered at round 0 the majority goes {n['six_maj0']:.3f} → {n['six_maj3']:.3f}. {n['fixes_tied']} of the fixes are items on which the correct answer was tied for the round-0 plurality and lost the tie-break; with ties split as fractional credit the round-0 vote is {n['split_maj0']:.3f}.")
    has(f"- Unanimously wrong among the answering agents at round 3: {n['unanimous_wrong_r3']} items ({', '.join(n['unanimous_wrong_r3_ids'])}).")
    has(f"- Mixed debate: of the {n['fixes_cut']} fixes on items with a cut-off round-0 row, the formerly cut-off agent is in the winning round-3 majority on {n['fixes_cut_joined']}; on the other {n['fixes_cut_not_joined']} it was still cut off or wrong at round 3. Items with a wrong round-0 majority: {n['wrong0_cut']} with a cut-off row, {n['wrong0_cut_fixed']} fixed; {n['wrong0_six']} with six answers, {n['wrong0_six_fixed']} fixed.")


def test_homogeneous_b5(out):
    h = out["debate"]["homogeneous"]
    for arm, r in h["arms"].items():
        a, b = r["r0"], r["r3"]
        has(f"| {arm} | {r['items']} | {a['agent_acc']:.3f} | {a['majority']:.3f} | {b['agent_acc']:.3f} | {b['majority']:.3f} | {a['unanimous']} | {b['unanimous']} | {a['unanimous_wrong']}→{b['unanimous_wrong']} | {a['all_wrong']} | {r['fixed']} | {r['broken']} | {a['phi']:.2f} | {b['phi']:.2f} |")
    c = h["common"]
    has(f"On the {c['items']} items all three arms completed: round-0 majority {c['maj0']:.3f}; round-3 majority " + ", ".join(f"{arm} {c[arm]['maj3']:.3f} ({c[arm]['gain']:+.3f}; fixed {c[arm]['fixed']} / broken {c[arm]['broken']})" for arm in ("plain", "selfonly", "pressure")) + ".")
    has("- Homogeneous debate, cut-off rows: " + "; ".join(f"{arm} {r['truncated_rows_r1_3']} of {r['rows_r1_3']} rows (rounds 1–3)" for arm, r in h["arms"].items()) + ".")


def test_aggregate(out):
    for mode, inp in (("synth", "6 debate round-3 answers"), ("moa", "5 own T=1 samples")):
        r = out["synthesis"][mode]
        has(f"| {mode} | {inp} | {r['items']} | {r['input_majority']:.3f} | {r['aggregator']:.3f} | {r['any_input']:.3f} | {r['none_correct_recovered']}/{r['none_correct_items']} | {r['novel']} | {r['novel_correct']} |")
