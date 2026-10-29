# budget-forcing

Budget-forcing protocol runs of DeepSeek V3.2 (STATS G curves): visible thinking budget B tokens + a second call for the answer.

- Built 2026-10-18 from our raw runs (not published); state of 2026-10-09.
- License / redistribution: our own model outputs (derived fields only). Item ids refer to MMLU-Pro (MIT), GPQA Diamond (CC BY 4.0, 'do not reveal online': only ids and 0/1 correctness here; no letters, texts or hashes), GSM-Hard (MIT), AIME/HMMT (see corpus-olympiad).
- Files: rows.csv — layer, id, budget, sample, correct (0/1), finish, out_tok. 4,262 rows (MMLU-Pro ~560 items x B 256..4096, GPQA 100, GSM-Hard 96, olympiad 100; excluded items dropped).
- Used in: v8 §10.2 (budget forcing vs native cut), cost-matched Q1 column 'budget-forcing'. Operator: `operators/budget.py` via `operators/p2parts/budget.py`.
