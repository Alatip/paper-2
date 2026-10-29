# Preregistration — paper 2, set B (generative operators)

This is a condensed English rendering of the preregistration kept in the authors' local version control. The original
was written on 2026-10-20 and amended on 2026-10-22, 2026-10-23 and 2026-10-26, each time before any operator call that
produced the scored data. Cross-references to internal working files and editorial notes are removed; the hypotheses,
thresholds, decision rule and amendments are reproduced without change in substance. Set A (the paper-1 corpus) is
exploratory: the hypotheses below were derived from it and are not tested on it.

## 1. Fixed before the operator runs

**Pool (as amended 2026-10-22).** Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.6, DeepSeek V3.2, GLM 4.7 Flash,
Qwen3.8-27B, Kimi K2.5. Reasoning effort medium; output cap 64,000 tokens (an answer that hits the cap is wrong);
provider-default temperature; one fixed prompt for all models.

**Items (as amended 2026-10-22).** 200 short-answer items: 170 medium (solved by Claude Haiku 4.5 at least once in
screening) and 30 hard (never solved by Claude Haiku 4.5; random sample, seed 0). The item list is fixed before any
operator call. A gold audit of items that no model solves may correct golds but does not change the item list.

**Answer identity.** As in paper 1 (12-significant-digit canonical form; a group of equal answers is correct if any member
is correct); ties go to the first participant in the fixed pool order.

**Operators and starting points.**
- Self-consistency: plurality of k = 5 draws (draw 1 + 4) of DeepSeek V3.2 and GLM 4.7 Flash; starting point — draw 1.
- Debate of the 7 models: 2 rounds after round 0; only items without unanimity in round 0 are debated (unanimous items
  keep the round-0 answer); each agent sees the other agents' final answers and the last 1,500 characters of their answer;
  result — plurality of round 2; starting point — plurality of round 0.
- Debate of copies: 5 copies of DeepSeek V3.2, round 0 = the 5 self-consistency draws, 2 rounds, same rules;
  starting points — draw 1 and the plurality of round 0.
- Synthesis: Claude Opus 4.6 reads the 7 round-0 answers (final answer + 1,500 characters each) and writes one answer;
  starting point — plurality of its inputs.
- Decomposition (split): plan of 2–4 subtasks, subtasks solved in order by the same model, assembly; on 120 items
  (all 30 hard + 90 medium, random, seed 0) for DeepSeek V3.2, GLM 4.7 Flash and Claude Sonnet 4.6; starting point —
  one call of the same model. Protocol as revised on 2026-10-23 (before data): the plan is generated without reasoning
  (Claude: effort low), every subtask and the assembly have the 64,000 cap; tokens are not equalised, so the comparison is
  at cost (below).

**Costs.** Provider billing for open models; Claude at API list price without cache. Operator cost includes round 0.

## 2. Analysis plan

- Δ = operator accuracy − starting-point accuracy on the same items; paired bootstrap over items (10,000 resamples,
  percentile 95 % CI); exact McNemar test on fixed vs broken items (W vs L).
- Held-out best and best(C(A)): best model (best model whose single answer costs no more than the operator) chosen on one
  half of the items and scored on the other, two-fold, 20 random splits stratified by tier and area, fixed seed.
- Multiplicity: Holm over the primary family; Benjamini–Hochberg (q = 0.10) over the secondary family; exploratory
  results without correction, marked as such.
- Decision rule: **confirmed** — predicted direction and the CI beyond the threshold; **refuted** — the CI entirely on the
  other side of the threshold; otherwise **undetermined**. "No effect" means the CI lies within ±3 points.
- Power: with 200 items the detectable Δ is about 5–6 points (about 7 on the 120 decomposition items); many outcomes are
  expected to be undetermined.

## 3. Hypotheses

Primary (Holm): H1a, H3, H4a, H4b, H7, H9. Secondary (BH): H1b, H2, H5, H6, H8. Exploratory: H10.

- **H1 Self-consistency plateau.** (a) For each model, plurality of k draws − mean single draw ≥ 0; on average > 0
  (lower CI bound > 0). (b) Plurality@9 − plurality@5 < 1 point (upper CI bound < 2).
- **H2 Self-consistency ≈ modal share.** |plurality@k − MAP over 16 draws| ≤ 3 points.
- **H3 Draws of one model are more correlated than different models.** Mean φ within a model's draws − mean φ between
  different models' answers > 0, lower CI bound > +0.10.
- **H4 Debate of different models.** (a) round final − round 0 > 0, lower CI bound > 0, W > L.
  (b) debate − held-out best ≤ +2 points (upper CI bound < +3).
- **H5 Panel collapse.** φ̄(final round) − φ̄(round 0) ≥ +0.20 (lower CI bound > 0); the oracle of the final round is
  below the oracle of round 0.
- **H6 Copies ≈ re-asking.** Δ(copies seeing each other) − Δ(copies seeing only their own answer) ∈ [−3, +3] points.
- **H7 Synthesis creates almost no new correct answers.** C⁺ ≤ 2 % of items (Clopper–Pearson upper bound ≤ 4 %) and
  Δ(synthesis − plurality of inputs) ∈ [−3, +3] points.
- **H8 Split decomposition does not solve the unsolvable.** On the hardest items C⁺ ≤ 2 %; Δ(split − single call) ≤ 0 on
  solvable items (upper CI bound < +3); team − split ≤ 0 (upper CI bound < +3).
- **H9 No operator beats best(C) at equal cost.** For every operator A, Δ(A − best(C(A))) has an upper CI bound < +3 points
  and no operator has a lower CI bound > 0.
- **H10 Gain grows with the prize** (exploratory). Collected share by tier: easier > harder.

## 4. Exclusions and stopping

The run covers the fixed item list; no items are added or removed after the list is fixed. Items whose gold is judged
defective by the audit are reported with and without them. An API error is retried up to 3 times; a missing answer is a
missing vote and is counted as wrong for accuracy. A spending cap is set before the run; reaching it stops the run, with
a report on the completed operators.

## 5. Decisions taken after the data (not preregistered)

- Gold audit of the 23 items no model solved: 1 gold corrected, 3 grader fixes, 3 defective items (kept; analysis without
  them reported as sensitivity).
- For resampling operators (self-consistency, copies) the collected share of the prize is replaced by the share of the
  pass@k headroom (pass@k − mean draw).
- Unanimity for selecting debated items was decided by the answer string (as preregistered); the analysis uses the paper-1
  answer identity, and the number of debated items that are unanimous under it is reported.
- Claude cost of calls cut at the 64,000 cap is estimated (lower bound).
- Grader: a bare Unicode root (e.g. 8√6) was parsed incorrectly; fixed on 2026-10-28, 56 grades changed from wrong to
  right; conclusions unchanged.
- 14 items (AIME 2024/2025, HMMT February 2025) are shared with the paper-1 corpus although the selection plan excluded
  AIME 2025 and HMMT February 2025; results without them are reported.
- 2026-10-28: the multiplicity family for operator-vs-starting-point McNemar tests is the 8 comparisons of the paper's
  tables 4.4, 4.5, 4.7 and 4.8 (Holm; BH alongside).
- 2026-10-29: an answer that reached the 64,000-token cap exactly (tok_out >= 64,000, not only > 64,000) is cut and
  counted as wrong (Claude Code CLI continued such answers in a second turn). Changed: 2 first answers of Claude Opus 4.6
  that had been graded correct (Opus 0.710 -> 0.700, oracle 0.905 -> 0.900); 36 other answers were relabelled "cut" but
  were already wrong. Conclusions unchanged.
- 2026-10-29: added after the main results of set B, before the run: self-consistency of Claude Opus 4.6 with 3 draws
  (draw 1 = the round-0 answer, 2 more draws with the same settings; 5 draws were not run for budget reasons).
  The comparison joins the Holm family, which grows from 8 to 9 tests.
