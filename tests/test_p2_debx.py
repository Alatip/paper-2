"""Regression tests for area "debx" (v8 §6.4–§6.7): the ported operators reproduce the frozen outputs of the original scripts
(tests/expected2/debx/) at the printed rounding. Each check rebuilds a printed line from run() output and looks it up verbatim.

Frozen files: phase1-out.md, phase2-out.md, tn5-tables-B2.md, axisH-RESULTS-raw.md and axisH-RESULTS-H6-raw.md. Known, documented difference: phase2 "synthesis picked an answer equal to DeepSeek's
own on 76 of 146" (the original compared GPQA letters with option hashes; the dataset holds letters, so it is 96 now).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "operators"))
EXP = ROOT / "tests" / "expected2" / "debx"

import debate_bounds  # noqa: E402
import debate_weak  # noqa: E402

f3 = lambda x: "—" if x is None else f"{x:.3f}"


@pytest.fixture(scope="module", autouse=True)
def _old_answer_rule():
    """The frozen outputs were produced with the v8 answer identity: replay them under P2_ANSWER_RULE=v8 (the default is
    paper 1's rule, tested in test_p2_answer_rule.py)."""
    from p2parts import answer_id as AID
    prev = AID.RULE; AID.set_rule("v8")
    yield
    AID.set_rule(prev)


@pytest.fixture(scope="module")
def weak(): return debate_weak.run()["debate_weak"]


@pytest.fixture(scope="module")
def bounds(): return debate_bounds.run()["debate_bounds"]


def _has(text, line):
    assert line in text, f"not found in frozen output:\n{line}"


# ----------------------------------------------------------------------------- weak pool, phase 2
def test_phase2_debate_tables(weak):
    t = (EXP / "phase2-out.md").read_text()
    d = weak["debate"]
    _has(t, f"Rows collected: {d['rows_collected']} of {d['rows_planned']}; items complete in all rounds: {d['complete']} of 196")
    _has(t, "strongest first = W sorted by round-0 accuracy on the completed contested items (in-sample): " + ", ".join(d["strong_order"])
         + " (" + ", ".join(f"{k} {v:.3f}" for k, v in d["strong_acc"].items()) + ")")
    for g in ("contested", "control", "allwrong", "all"):
        b = d["groups"][g]
        _has(t, f"{b['n']} completed problems (" + ", ".join(f"{l} {c}" for l, c in b["layers"].items()) + ")")
        for rd, r in enumerate(b["rounds"]):
            _has(t, f"| {rd} | {r['agent_acc']:.3f} | {r['plur_first']:.3f} | {r['plur_strong']:.3f} | {r['plur_split']:.3f} | {r['ties']} | {r['unanimous']} | "
                    f"{r['fixed_first']} / {r['broken_first']} | {r['fixed_strong']} / {r['broken_strong']} | {f3(r['mean_phi'])} ({r['phi_pairs']}/6) | "
                    f"{f3(r['n_eff'])} | {r['all_wrong']} | {r['unanimous_wrong']} | {r['cutoff_rows']} |")
        _has(t, f"Round 0 → 3 gain: first in order {b['gain_first']:+.3f} ({b['fixed_first']} fixed / {b['broken_first']} broken, exact McNemar p = {b['p_first']:.3f}); "
                f"strongest first {b['gain_strong']:+.3f} ({b['fixed_strong']} / {b['broken_strong']}, p = {b['p_strong']:.3f}); ties split {b['gain_split']:+.3f}. "
                f"Fixed problems whose round-0 plurality was a tie: {b['fixed_first_tie_r0']} of {b['fixed_first']} (first in order).")
        for layer, v in b["per_layer"].items():
            _has(t, f"{layer} ({v['n']}): " + " / ".join(f"{v[k + '_r0']:.3f} → {v[k + '_r3']:.3f}" for k in ("pf", "ps", "pe")))


def test_phase2_members_projection_flips(weak):
    t = (EXP / "phase2-out.md").read_text(); d = weak["debate"]
    for m, v in d["members"].items():
        _has(t, f"| {m} | {v['acc_r0']:.3f} | {v['first_lost']} / {v['first_won']}, p = {v['p_first']:.3f} | {v['strong_lost']} / {v['strong_won']}, p = {v['p_strong']:.3f} |")
    p = d["projection"]
    _has(t, f"the {p['allright_undebated']} all-W-right problems outside the control were not debated): round-0 W plurality {p['r0']:.3f}; "
            f"round-3 {p['r3_nobreak']:.3f} if none of the undebated all-right problems breaks, {p['r3_break']:.3f} if they break at the control's rate "
            f"({p['control_broken']} of {p['control_n']}, Wilson [{p['control_wilson'][0]:.3f}, {p['control_wilson'][1]:.3f}])")
    _has(t, "share that flipped to wrong: " + "; ".join(f"{k} peers: {v['flipped']}/{v['cases']} ({v['share']:.2f})" for k, v in d["flips"].items()) + ".")
    _has(t, "Mean completion tokens per agent-turn (rounds 1–3): " + ", ".join(f"{k} {v:.0f}" for k, v in d["tokens_mean"].items()) + ".")


def test_phase2_synth(weak):
    t = (EXP / "phase2-out.md").read_text(); s = weak["synth"]
    for g, lab in (("contested", "contested-for-W"), ("allwrong", "all W wrong")):
        v = s[g]
        _has(t, f"- {lab}: {v['n']} items; synthesis correct on {v['correct']} ({v['acc']:.3f}, Wilson [{v['wilson'][0]:.3f}, {v['wilson'][1]:.3f}]); "
                f"unparsed {v['unparsed']}; novel answers (not among the four inputs) {v['novel']}, of them correct {v['novel_correct']}.")
    c = s["contested"]
    for k, nm in (("first", "first in order"), ("strong", "strongest first")):
        v = c[f"vs_plur_{k}"]
        _has(t, f"Round-0 W plurality ({nm}) {v['plur']:.3f} vs synthesis {c['acc']:.3f}: plurality right & synthesis wrong / reverse {v['plur_right_synth_wrong']} / {v['reverse']}, p = {v['p']:.3f}.")
    _has(t, f"DeepSeek's own round-0 answer on these items: {c['deepseek_own_acc']:.3f}")
    _has(t, f"round-3 W plurality (strongest first) {c['debate_r3_strong_on_both']:.3f} vs synthesis {c['synth_on_both']:.3f}.")
    _has(t, f"Synthesis rows {s['rows']}, OpenRouter cost ${s['cost']:.2f}.")
    assert c["equal_own"] == 96   # documented: 76 in the frozen output (letter vs hash on GPQA)


def test_phase2_judge(weak):
    t = (EXP / "phase2-out.md").read_text(); j = weak["judge"]
    for k, v in j["judges"].items():
        _has(t, f"| {k} | {v['pick_acc']:.3f} | {v['named']} | {v['named_acc']:.3f} | {v['own_acc']:.3f} | {v['picked_own']:.3f} | {v['picked_own_when_wrong']:.3f} ({v['own_wrong_n']}) | {v['none_on_allwrong']} / {j['n_allwrong']} |")
    _has(t, f"Mean pairwise φ of 'pick correct' over the {j['n']} items: {f3(j['mean_phi'])} (3 pairs; tetrachoric {f3(j['mean_tetra'])}); n_eff of 3 judges {f3(j['n_eff'])}.")
    for k, v in j["tie_rules"].items():
        tail = f"{v['won']} / {v['lost']}, p = {v['p']:.3f}" if k != "split" else "–"
        _has(t, f"| {v['solvers']:.3f} | {v['panel']:.3f} | {v['diff']:+.3f} | {tail} |")
    st = j["strict"]
    _has(t, f"answers {st['answered']} of {j['n']}; right on {st['acc']:.3f} [Wilson {st['wilson'][0]:.3f}, {st['wilson'][1]:.3f}]; W plurality (strongest first) on those: {st['plur_on_answered']:.3f}; on the {j['n'] - st['answered']} abstained: {st['plur_on_abstained']:.3f}.")
    v = j["vs_debate"]
    _has(t, f"on the {v['n']} items with both: {v['panel']:.3f} vs {v['debate']:.3f}; won / lost {v['won']} / {v['lost']}, p = {v['p']:.3f}.")
    w = j["whole_733"]
    _has(t, f"W plurality {w['plur_first']:.3f} (first in order) / {w['plur_strong']:.3f} (strongest first); W plurality + J3 {w['panel_first']:.3f} (first-judge ties; empty panel = wrong) / "
            f"{w['panel_fallback']:.3f} (tied or empty panel falls back to the plurality) / {w['panel_strong']:.3f} (strongest-judge ties); best single W member in-sample "
            f"{w['best_member']} {w['best_member_acc']:.3f}; DeepSeek fixed {w['deepseek_acc']:.3f}; oracle over W {w['oracle']:.3f}.")
    _has(t, f"Judge rows {j['rows']}; cost ${j['cost']:.2f}")


def test_phase2_spend(weak):
    t = (EXP / "phase2-out.md").read_text(); s = weak["spend"]
    for k in ("debate-w", "synth-w", "judge-w"):
        v = s[k]; _has(t, f"| {k} | {v['rows']} | {v['openrouter']:.2f} | {v['haiku_cli']:.2f} | {v['openrouter'] + v['haiku_cli']:.2f} |")
    _has(t, f"| total | 2976 | {s['total']['openrouter']:.2f} | {s['total']['haiku_cli']:.2f} | {s['total']['all']:.2f} |")


# ----------------------------------------------------------------------------- weak pool, phase 1 (a): W inside the six-agent debate
@pytest.mark.parametrize("key,npairs", [("A6", 15), ("W", 6), ("W3", 3), ("W_allanswered", 6), ("W3_allanswered", 3)])
def test_phase1_six_tables(weak, key, npairs):
    t = (EXP / "phase1-out.md").read_text(); b = weak["six"][key]
    _has(t, f"{b['n']} problems; pool order")
    _has(t, "strongest-first order " + ", ".join(b["strong_order"]) + " (round-0 accuracy with a cut-off scoring 0: " + ", ".join(f"{k} {v:.3f}" for k, v in b["strong_acc"].items()) + ")")
    fx = [0, 0, 0, 0]
    for rd, r in enumerate(b["rounds"]):
        _has(t, f"| {rd} | {r['agent_acc']:.3f} | {r['plur_first']:.3f} | {r['plur_strong']:.3f} | {r['plur_split']:.3f} | {r['ties']} | {r['unanimous']} | ")
        _has(t, f"| {f3(r['mean_phi'])} ({r['phi_pairs']}/{npairs}) | {f3(r['n_eff'])} | {r['all_wrong']} | {r['unanimous_wrong']} | {r['cutoff_rows']} |")
    _has(t, f"Round 0 → 3, first in order: {b['fixed_first']} fixed / {b['broken_first']} broken, exact McNemar p = {b['p_first']:.3f}; strongest first: {b['fixed_strong']} / {b['broken_strong']}, "
            f"p = {b['p_strong']:.3f}; ties split: {b['gain_split']:+.3f}. Problems with a pool agent cut off at round 0: {b['cut0']} of {b['n']}; fixes on them: {b['fixed_first_on_cut0']} of {b['fixed_first']} "
            f"(first in order), {b['fixed_strong_on_cut0']} of {b['fixed_strong']} (strongest first). Fixed problems where the round-0 plurality was a tie: {b['fixed_first_tie_r0']} of {b['fixed_first']}.")


def test_six_flips(weak):
    t = (EXP / "tn5-tables-B2.md").read_text()
    for k, v in weak["six"]["flips6"].items():
        _has(t, f"| {k} | {v['cases']} | {v['share']:.2f} |")


# ----------------------------------------------------------------------------- bounds
def test_h3(bounds):
    t = (EXP / "axisH-RESULTS-raw.md").read_text(); h = bounds["h3"]
    import ast, re
    m = re.search(r"Items: (\d+) \((\{.*?\})\)\.", t)
    assert int(m.group(1)) == h["n"] and ast.literal_eval(m.group(2)) == h["layers"]   # same counts; dict order differs
    for c, v in h["by_c"].items():
        m = v["maj"]; lo, hi = v["r3_wilson"]
        _has(t, f"| {c} | {v['n']} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | {m[3]:.2f} | [{lo:.2f}, {hi:.2f}] | {m[3] - m[0]:+.2f} |")
    _has(t, "Crossover (any c with r3 majority below r0 majority, Wilson upper bound below r0): " + (str(h["crossover"]) if h["crossover"] else "none") + ".")
    for t_ in range(3):
        cells = [("—" if not h["flips"][f"{t_}->{t_ + 1}.m{m}"]["cases"] else f"{h['flips'][f'{t_}->{t_ + 1}.m{m}']['share']:.2f} ({h['flips'][f'{t_}->{t_ + 1}.m{m}']['cases']})") for m in range(6)]
        _has(t, f"| {t_}→{t_ + 1} | " + " | ".join(cells) + " |")
    _has(t, f"Element error at round 0 (mean agent error): {h['element_error_r0']:.3f}; share of items with ≥ 4 of 6 correct at r0: {h['share_ge4_r0']:.3f}.")


def test_h6c(bounds):
    t = (EXP / "axisH-RESULTS-H6-raw.md").read_text(); h = bounds["h6c"]
    for layer in ("mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad", "pooled"):
        v = h[layer]
        cells = [f"{v[k]['maj']:.3f} ({v[k]['delta']:+.3f}) [{v[k]['ci'][0]:+.3f}, {v[k]['ci'][1]:+.3f}] n={v[k]['n']}" for k in ("plain", "selfonly", "pressure", "mixed")]
        _has(t, f"| {layer} | {v['n']} | {v['map']:.3f} | {v['single']:.3f} | " + " | ".join(cells) + " |")
