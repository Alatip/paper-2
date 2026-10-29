"""Answer identity and plurality for every paper-2 operator (decided 2026-10-18).

Default rule RULE = "p1": EXACTLY paper 1's rule (paper-1 text, section 2 "Voting"; lib/p1stats/data.same_answer +
voting.answer_groups): fractions evaluated, numbers keyed at 12 significant digits, integers exact; two decimals are one
answer if the longer, rounded half-up to the shorter's decimals, equals it; groups are built by chaining (transitive closure)
over the answers of ONE vote; expressions normalised (\\frac -> a/b, \\sqrt -> √, no spaces / '$').
A group (one "answer") is correct if any of its members is graded correct (paper 1 voting.plurality: `right` = groups with a
correct vote).

Old rule RULE = "v8": each operator's own v8 identity (tech-note-5 canon with 7 significant digits; raw strings in
debate_bounds) and the v8 "first row of the winning answer decides" correctness. Kept so the
effect of the change can be shown and so the regression tests against the frozen v8 outputs still
verify the port. Select with the environment variable P2_ANSWER_RULE=v8 (read at import) or set_rule("v8").

Tie rules are NOT changed here: each operator keeps its v8 tie rule ("first": the tied answer whose first vote comes first in
the fixed order wins -- identical to paper 1's rule 'first'; "split": 1/k credit -- identical to paper 1's 'split'; budget: a tie is no answer / wrong -- not a paper-1 rule).
"""
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from p1stats import voting as _V  # noqa: E402

RULE = os.environ.get("P2_ANSWER_RULE", "p1")
assert RULE in ("p1", "v8"), RULE


def set_rule(rule):
    global RULE
    assert rule in ("p1", "v8"), rule
    RULE = rule


def keys(answers, old_key=lambda a: a):
    """Identity key per answer of ONE vote (list; None for an empty answer). p1: group id (index of the group's first
    member) from paper-1 answer_groups; v8: old_key(answer)."""
    answers = list(answers)
    if RULE == "v8":
        return [None if a in (None, "") else old_key(a) for a in answers]
    return _V.answer_groups(answers)


def vote(answers, oks=None, old_key=lambda a: a):
    """Plurality of one vote, v8 tie rule 'first' (Counter.most_common keeps first-seen order = paper 1 'first').
    Returns (top_key, count, correct, keys). correct: p1 -> any member of the winning group correct; v8 -> the first row
    with the winning key decides (tech-note-5 maj_correct). No answers -> (None, 0, False, keys)."""
    answers = list(answers)
    k = keys(answers, old_key)
    c = collections.Counter(x for x in k if x is not None)
    if not c:
        return None, 0, False, k
    top, cnt = c.most_common(1)[0]
    if oks is None:
        return top, cnt, None, k
    oks = list(oks)
    if RULE == "v8":
        ok = bool(next(o for x, o in zip(k, oks) if x == top))
    else:
        ok = any(bool(o) for x, o in zip(k, oks) if x == top)
    return top, cnt, ok, k


def right_keys(k, oks):
    """Keys that count as correct. p1: any member correct; v8: the key's first row decides."""
    if RULE == "v8":
        first = {}
        for x, o in zip(k, oks):
            if x is not None and x not in first: first[x] = bool(o)
        return {x for x, o in first.items() if o}
    return {x for x, o in zip(k, oks) if x is not None and o}
