# Set B analysis — specification of `operators/p2parts/setb.py`

Contract for blind tests (tests are written from this file and the signatures only, never from the implementation).
Module: `from p2parts import setb as SB` with `operators/` on `sys.path`. Pure functions: no file I/O; randomness only through
explicit seeds / `numpy.random.Generator` arguments. Constants: `SB.CAP = 64000`, `SB.NBOOT = 10000`, `SB.NSPLIT = 20`.

Answer identity used below is paper 1's rule, provided by `p2parts.answer_id` with `set_rule("p1")`: within ONE vote, answers
are grouped by `answer_id.keys(answers)` (fractions evaluated, numbers at 12 significant digits, integers exact, chained
groups); `None` means "no answer" and gets key `None`. Tests may rely on these facts of the p1 rule: identical strings share a
key; `"1/2"` and `"0.5"` share a key; `"2"` and `"3"` do not; `None` -> `None`.

## Answers

### `norm_answer(s: str | None) -> str`
Normalises a LaTeX-like answer for explicit-equivalent matching.
1. `None` -> `""`.
2. Remove, everywhere: every `$`, every whitespace character, the commands `\displaystyle`, `\left`, `\right`, `\!`, `\,`,
   `\;`, and the delimiters `\(` and `\)`; THEN remove one trailing `.` of the result (so `"$x = 1.$"` -> `"x=1"`).
3. Replace `\dfrac` and `\tfrac` by `\frac`; replace `\overline{R}` by `\overlineR`.
4. If the result starts with `\Gamma(f)=`, remove that prefix.
Examples: `"$\dfrac{1}{5}$"` -> `"\frac{1}{5}"`; `"\arcsin\!\left(\frac{1}{5}\right)"` -> `"\arcsin(\frac{1}{5})"`;
`"x = 2."` -> `"x=2"`.

### `apply_audit(items: dict, audit: dict) -> (dict, dict)`
`items`: `{id: row-dict}`. `audit`: may contain `"relabel"` and `"grader_fix"`, each `{id: patch-dict}`; other keys ignored.
Returns `(new_items, accept)`:
- `new_items` is a NEW dict (the input dict and its rows are not mutated). For every id present in `items` and in either
  patch mapping, the row becomes a copy of the old row updated with the patch's `"gold"` and `"gold_norm"` keys only (other
  patch keys such as `"why"`, `"accept"` are NOT copied into the row). Ids in a patch but not in `items` are ignored. Rows of
  other ids are unchanged. If an id is in both mappings, both patches apply (grader_fix after relabel).
- `accept`: `{id: set(norm_answer(x) for x in patch["accept"])}` for every `grader_fix` patch with a non-empty `"accept"`
  list, whether or not the id is in `items`; ids without such a list are absent.

### `answer_hash(s) -> str`
`sha256(norm_answer(s).encode("utf-8")).hexdigest()`.

### `is_correct(row, grade_fn, accept=None, parse_fn=None, cap=64000, hashed=False) -> int`
`row`: dict or `None`. Returns 0 or 1.
- 0 if `row` is falsy, or `row.get("finish")` is `"length"` or `"error"`, or `row.get("tok_out") or 0` >= `cap` (rule of 2026-10-29; before it `> cap`).
- Else, if `accept` is not `None`: let `a = parse_fn(row.get("text") or "")`; 1 iff `norm_answer(a)` is in `accept`
  (`hashed=False`) or `answer_hash(a)` is in `accept` (`hashed=True`); `grade_fn` is NOT called.
- Else: `int(bool(grade_fn(row)))`.

### `plurality(answers: list, oks: list) -> (int, key, list)`
One vote. `answers[i]` is a string or `None`/`""` (no answer; `""` is treated as `None`); `oks[i]` is 0/1.
Groups: `answer_id` keys of the answers. The winning group is the one with the most members; on a tie the group whose first
member appears earliest wins. Returns `(correct, top_key, keys)`: `correct` = 1 iff ANY member of the winning group has
`oks == 1`; `top_key` = winning key; `keys` = key per answer. If no answer is present: `(0, None, keys)`.

### `is_unanimous_by_identity(keys: list) -> bool`
True iff exactly one distinct non-`None` key occurs in `keys` (an all-`None` list is NOT unanimous).

### `new_correct_answer(op_answer, op_ok, input_answers) -> int`
1 iff `op_ok` is truthy, `op_answer` is a non-empty string, and in the joint grouping
`answer_id.keys(input_answers + [op_answer])` the operator's key differs from every input's key. Else 0.

### `plurality_codes(codes: list, oks: list) -> (int, code | None)`
Plurality over opaque answer codes (public data): equal non-empty codes = one answer group; `""` or `None` = no answer.
Winner = the code with the most members; on a tie the code whose first occurrence comes earliest. Returns
`(correct, winner)`: `correct` = 1 iff any position with the winning code has `oks == 1`. No non-empty code -> `(0, None)`.

### `new_correct_code(op_code, op_ok, input_codes) -> int`
1 iff `op_ok` is truthy, `op_code` is non-empty, and `op_code` is not in `input_codes`; else 0.

## Cost

### `row_cost(row, model: str, list_prices: dict, cap=64000) -> (float, bool)`
`$` of one call and whether it is an estimate.
- `row` falsy -> `(0.0, False)`.
- `model` not starting with `"claude-"`: `(float(row.get("cost") or 0), False)`.
- `model` starting with `"claude-"`: if `row.get("cost_notional")` is not `None` and not `""` -> `(float(that), False)`;
  else `(pi_in * float(row.get("tok_in") or 0) + pi_out * cap, True)` with `(pi_in, pi_out) = list_prices[model]`.

## Statistics

### `operator_seed(name: str) -> int`
`int(sha256(name.encode()).hexdigest()[:8], 16)`.

### `bootstrap_ci(d, rng, n_boot=10000) -> [mean, lo, hi]`
`d`: 1-D numeric array of length n. Draws ONE matrix `rng.integers(0, n, size=(n_boot, n))`, takes row means of `d` at those
indices, returns `[mean(d), percentile 2.5, percentile 97.5]` (numpy default linear percentiles) as Python floats. Consumes
`rng` (two calls with the same rng give different resamples; same seed -> same result).

### `splits(strata: list, n_split=20, seed=0) -> list[np.ndarray[bool]]`
`n_split` boolean masks of length `len(strata)`. One `rng = numpy.random.default_rng(seed)` for all masks. For each mask, for
each distinct label in `sorted(set(strata))`: take the indices with that label (ascending), shuffle them with `rng.shuffle`,
set the first `floor(size/2)` of the shuffled indices to True. Properties: within each stratum exactly `floor(size/2)` Trues;
deterministic for a given seed.

### `heldout_best(P, S) -> np.ndarray`
`P`: items × participants, 0/1. For every mask `m` in `S` and both directions `(choose, score) in ((m, ~m), (~m, m))`: pick
the participant column `j = argmax` of the column means of `P[choose]` (first index on ties) and add `P[score, j]` to the
score items. Return the per-item sum divided by `len(S)` (each item is scored exactly once per mask).

### `heldout_best_under_cost(acc, cost, CA, S, names) -> (np.ndarray | None, Counter)`
`acc`, `cost`: `{name: per-item array}`; `CA`: per-item operator cost; `names`: ordered candidate names. For every mask and
both directions as above: candidates = names with `mean(cost[name][choose]) <= mean(CA[choose])`; if none -> return
`(None, picks_so_far)` immediately; else pick the maximum of the tuple `(mean acc on choose, -mean cost on choose, name)` and
add `acc[pick][score]` to the score items; count the pick. Return `(per-item sum / len(S), picks Counter)`.

### `resample_headroom(P, acc: float) -> dict`
`P`: items × draws, 0/1. `mean_draw` = mean of all entries; `pass_at_k` = mean over items of the row max;
`headroom = pass_at_k - mean_draw`; `share = (acc - mean_draw) / headroom` if `headroom > 0` else `None`.
Returns `{"mean_draw", "pass_at_k", "headroom", "share"}` (floats / None).

### `operator_stats(name, a, a0, P, CA, C0, acc, cost, strata, names, mcnemar, resample=False) -> (dict, Generator)`
All arrays aligned by item, length n. `a`, `a0`: operator / starting point 0/1; `P`: n × participants 0/1; `CA`, `C0`: `$` per
item; `acc`, `cost`: as above; `strata`: label per item; `mcnemar(W, L) -> p`.
`rng = numpy.random.default_rng(operator_seed(name))`; `S = splits(strata)` (seed 0).
Order of rng use: first `delta = bootstrap_ci(a - a0, rng)`, then (if defined) the best(C) gain CI.
Output keys:
- `N` = n; `acc` = mean a; `acc0` = mean a0; `delta`; `W` = #(a0 = 0, a = 1); `L` = #(a0 = 1, a = 0);
  `mcnemar_p = float(mcnemar(W, L))`; `C_usd` = mean CA; `C0_usd` = mean C0;
  `usd_per_extra_item` = `(mean CA - mean C0) / delta[0]` if `delta[0] > 0` else `None`.
- if `resample`: `resample = resample_headroom(P, mean a)`.
- if `P` has more than one column: `Cplus` = #(row max of P = 0 and a = 1); `oracle` = mean of row max; `best_heldout` = mean
  of `heldout_best(P, S)`; `best_insample` = max column mean; `prize = oracle - best_heldout`;
  `s_heldout = (acc - best_heldout) / prize` if `oracle > best_heldout` else `None`. With one column these keys are absent.
- `bestC_heldout`: `(v, picks) = heldout_best_under_cost(acc, cost, CA, S, names)`; `None` if `v` is `None`, else
  `{"acc": mean v, "picks": dict(picks), "gain": bootstrap_ci(a - v, rng)}`.
Returns `(out, rng)` — the same generator object, advanced, for further CIs of the same operator.
