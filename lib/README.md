# p1stats — statistics library shared with paper 1

`lib/p1stats/`, numpy + scipy only. The same library is published with paper 1 (https://github.com/Alatip/paper-1); the
"Paper" column below refers to sections of paper 1. Tests: `python3 -m pytest tests -q` (from the repository root)
(`tests/conftest.py` puts `lib/` on `sys.path`).

| Module.function | What | Paper |
|---|---|---|
| data.canon | canonical answer identity: numbers as exact values keyed at 12 significant digits (rule of 2026-10-17, see below) | §2 plurality, §4.3 certification strict |
| data.canon_v1 | OLD identity (tech-note-5 `stats.canon`, 7 significant digits); only for replaying frozen outputs and old-vs-new comparison | — |
| data.load_layer / correctness_matrix | corpus from datasets/corpus-*/ (answers.csv, run-order.csv), label-audit exclusion, all-7-present filter; correctness_matrix raises ValueError on `correct = None`; the matrix is in id-sorted order: fine for deterministic computations and for operators/factor (the original factor_corpus convention); for split-half or any RNG over items use the load_layer order (`sort_ids=False`) | §4.1 |
| metrics.accuracy, oracle, best, best_index, prize, collected_share, all_wrong | acc, coverage/pass@n, in-sample best (tie → first in pool order), prize = oracle − best, s = (acc − best)/(oracle − best) | §2 |
| metrics.split_half, heldout_best_minus, pct_2000 | split-half held-out best vs operator, `random.Random(0)`, 2,000 splits, CI = sorted[50], [1949]; `reshuffle='cumulative'`, `interval='numpy'` = the SWE window convention | §4 protocol, §4.2 |
| metrics.routing_in_sample, routing_split_half | per-set router ceiling, in-sample and held-out | §3.3, table 4.3a |
| metrics.router_stochastic | R*, expected one-run oracle with random answers | §3.3 |
| metrics.median_vs_heldout_best | median-member vs held-out best (Chen/Leaderboard bands) | §3.4 |
| correlation.phi, mean_phi | φ, φ̄ (undefined pairs dropped) | §2 |
| correlation.tetra_cospi, tetra_exact, mean_tetra | tetrachoric: cos-π (used by all scripts) and exact bivariate-normal (new, not used by scripts); empty 2×2 cell → `rule` (default `TETRA_RULE='haldane'`: +0.5 to every cell; `drop` = old convention, pair dropped; `limit` = ±1; only for the App. C sensitivity table); a constant vector stays undefined | §2, App. D |
| correlation.n_eff | Kish n/(1+(n−1)φ̄) | §2 |
| correlation.pair_matrix, subset_mean | pairwise statistic matrix and its mean over many subsets (public-band subsets) | §4.1, §5.2 |
| data.same_answer, decimal_round_match | pairwise answer identity of plurality: canon key OR decimal-rounding rule (2026-10-17, below) | §2 |
| voting.answer_groups | group id per answer: union-find closure of same_answer over one item; used by plurality, judges `--panel-key canon`, certification strict | §2, §4.3 |
| voting.plurality, plurality_rows | plurality on answer groups (default) or on key(answer); ties `first` (pool/strongest order) or `split` | §2 |
| voting.majority_correct | strict majority by correctness | §2 |
| voting.condorcet_homogeneous, poisson_binomial_pmf, condorcet_heterogeneous | Condorcet (binomial tail / Poisson–binomial) | §3.4 |
| voting.copula_conditional, copula_majority, copula_all_wrong | one-factor Gaussian copula, midpoint rule on [−6,6], K=400 | §3.4, §4.2 |
| voting.copula_many | the same, vectorised over pools (majority and all-wrong); `grid='hermite'` = Gauss–Hermite nodes (public-band script) | §3.4, table 4.2 |
| voting.median_member | ρ = 1 limit; even n: lower of the two middle members (`mean` = old convention, regression only) | §3.4 |
| tests_stats.mcnemar_exact | exact two-sided McNemar from b, c | §4 |
| tests_stats.paired_bootstrap_ci | item bootstrap of paired mean difference | §4 |
| tests_stats.holm, benjamini_hochberg | adjusted p-values | §4.3, App. D |
| tests_stats.wilson | Wilson interval | table 4.7 |
| factor.mp_edge, eigen_check | Marchenko–Pastur λ+ = (1+√(n/N))², row-permutation null (95th pct of max eigenvalue) | §2, §4.2 |
| factor.fit, eap_theta, lz | 2PL / Rasch MML-EM (verbatim from the IRT script of an earlier experiment of the series) | App. D |
| factor.residual_correlation, mean_offdiag_corr, q3_bias, drop_constant_items | mean residual r, bias −1/(n−1), excess | §4.2, App. D |
| factor.cv_persons | 5-fold CV over members, 2PL vs Rasch (held-out EAP and marginal log-likelihood per response) | §5.2, App. D |
| factor.family_stats, icc_by_family | family excess T (within − between) with permutation p, ICC | App. D |

## Convention notes (deliberately kept as in the old scripts)
- Two "median member" conventions coexist: `copula_major(ρ≥0.999)` takes sorted-desc[(n+1)//2−1] (upper median for even n),
  while the "median" column of table42/factor_corpus uses `statistics.median` (mean of the two middle values) for even n.
- `copula_majority` clamps p to [1e−9, 1−1e−9]; `copula_all_wrong` does not.
- Wilson with n = 0: `analyze.wilson` → (nan, nan) (kept); `stats.cp_interval` → (0, 0).
- cos-π tetrachoric is biased upward when margins are skewed (≈ +0.05 at ρ = 0.6 in the unit test); `tetra_exact` is provided for comparison only.
- Reproducing RNG-dependent numbers requires the same stream order (see `test_factor_eigen_and_irt`).

## Answer identity: canon() changed on 2026-10-17 (author's decision)
Old rule (`canon_v1`, copy of tech-note-5 `stats.canon`): numbers compared as floats rounded to 7 significant digits, fractions
not evaluated. It merged different GSM-Hard answers (50627678 vs 50627679, 72768591 vs 72768592, 26073328 vs 26073332,
42159089/3 vs 14053030) and kept 1/2 ≠ 0.5.
New rule (`canon`): after the same cleanup ('$', spaces, `\( \)`, thousands commas, trailing '.', `\frac{a}{b}`/`\dfrac` → a/b)
an answer that is a plain number, a decimal/scientific literal or a/b is parsed into an exact `Fraction`; the key is that value
rounded half-even to 12 significant digits, written as `d.dddddddddddE±x` (`0` for zero). So 1/2 = 0.5 = 5e-1 = `\frac{1}{2}`,
24112830.0000041 = 24112830, but 50627678 ≠ 50627679 and 9099577.92 ≠ 9099577.91666 (intended: they differ at 12 digits).
Implementation choice (not in the author's wording): an exact integer longer than 12 digits is keyed by all its digits, so two
23-digit integers that differ only at the end stay distinct. Non-numeric answers keep the text normalisation; letters unchanged;
None / '' kept. Paper 1's tests that replay frozen outputs (in the paper-1 repository) use the `canon_v1` fixture (tests/conftest.py).

## Decimal-rounding rule for votes (author's decision 2026-10-17, refines the 12-digit canon)
Two DECIMAL answers count as one vote if the one with more decimal places, rounded half-up (decimal arithmetic) to the number
of decimal places of the other, equals the other: 48179.43, 48179.42857 and 48179.42857142857 are one vote; 9099577.92 and
9099577.91666 are one vote. Integers are compared exactly and never merge with a decimal by rounding (48179 vs 48179.43 and
50627678 vs 50627679 stay different). Fractions a/b and scientific literals: exact value only (the canon key). Letters, text,
None: unchanged. `canon()` stays the key of exact identity; the vote relation is `data.same_answer` = same canon key OR
`data.decimal_round_match`. Because that relation is not transitive, `voting.answer_groups` takes its transitive closure
(union-find) over one item's answers. **Chaining caveat**: 0.06 ~ 0.1 and 0.14 ~ 0.1, so 0.06, 0.1 and 0.14 end up in one
group although 0.06 and 0.14 are not directly related; the partition does not depend on answer order. Used by
`voting.plurality` (default). Frozen replays (fixture `canon_v1`) swap in `data.same_answer_v1` (canon_v1 key only).

Conventions to keep in mind. Tetrachoric ρ: `p1stats.correlation` applies the Haldane +0.5 rule to an empty 2×2 cell (default), while the set-A loader `operators/p2parts/tn5.tetra` drops such a pair; none of the paper's ρ values comes from the latter. `operators/budget.py` picks best(C) in-sample (documented in its docstring); its outputs are not used in paper 2, where best(C) is held-out.
