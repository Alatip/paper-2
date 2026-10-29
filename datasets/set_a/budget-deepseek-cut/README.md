# budget-deepseek-cut

DeepSeek V3.2 native traces: one answer at T = 0 ('single') and 16 draws at T = 1 ('samples') per item, with correctness after a cut at B tokens.

- Built 2026-10-18 from our raw runs (not published); state of 2026-10-09.
- License / redistribution: our own model outputs (derived fields only). Item ids refer to MMLU-Pro (MIT), GPQA Diamond (CC BY 4.0, 'do not reveal online': only ids, GOLD/W labels and option letters (`akey`) here; no texts or hashes), GSM-Hard (MIT), AIME/HMMT (see corpus-olympiad).
- Files: `cut-correct.jsonl.gz` — the run's own text-free export (copied unchanged, gzip): per trace layer, stage, id, sample, out_tok, finish, correct, cost_usd and, for every B in {68..32768}, lower/upper correctness, is_cut and an anonymised vote label (GOLD/TIE/W*). 4,773 traces. `natural.csv` — the same traces uncut: layer, stage, id, sample, correct, finish, out_tok, cost_usd, vote. 4,773 rows (single 566/197/144/90; samples 944/960/912/960). `natural.uncapped.csv` — the same as `natural.csv` with answers cut at 8,192 output tokens replaced by the 65,536-token rerun.
- `vote`: GOLD = graded correct by our grader (audited gold, 2026-10-08); W1, W2, ... = distinct wrong answers of that item (per item, order of first appearance); empty = no parsable answer. No answer text or letters are stored.
- Used in: v8 §10.2–10.4, 10.6, table 10.1 (cost-matched Q1, Q2, Q3, Q5). Operator: `operators/budget.py` via `operators/p2parts/budget.py`.


## Answer keys (added 2026-10-18)

`akey` = answer key for paper 1's answer-identity rule (operators/p2parts/answer_id.py): MMLU-Pro and GPQA rows hold the option letter our grader parses, or an opaque per-item code `~k` when no letter parses; other layers hold the parsed answer string. Empty = no answer. Added 2026-10-18 from the same raw runs; every earlier column is unchanged.
