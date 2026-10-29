# budget-debates

Three-round debates: six DeepSeek V3.2 copies ('copies', debate-homo-plain) and six model families ('families': DeepSeek V3.2, Claude Haiku 4.5, Claude Sonnet 5.5, Qwen 3.7 Plus, GLM 5.3 Flash, Mistral Large 2512).

- Built 2026-10-18 from our raw runs (not published); state of 2026-10-09.
- License / redistribution: our own model outputs (derived fields only). Item ids refer to MMLU-Pro (MIT), GPQA Diamond (CC BY 4.0, 'do not reveal online': only ids, GOLD/W labels and option letters (`akey`) here; no texts or hashes), GSM-Hard (MIT), AIME/HMMT (see corpus-olympiad).
- Files: rows.csv — layer, kind, id, round, agent, finish, out_tok, vote. 8,335 rows. `rows.uncapped.csv` — the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun.
- `vote`: GOLD = graded correct by our grader (audited gold, 2026-10-08); W1, W2, ... = distinct wrong answers of that item (per item, order of first appearance); empty = no parsable answer. No answer text or letters are stored.
- Used in: v8 §10.4 (operators at own cost), cost-matched Q3. Operator: `operators/budget.py` via `operators/p2parts/budget.py`.


## Answer keys (added 2026-10-18)

`akey` = answer key for paper 1's answer-identity rule (operators/p2parts/answer_id.py): MMLU-Pro and GPQA rows hold the option letter our grader parses, or an opaque per-item code `~k` when no letter parses; other layers hold the parsed answer string. Empty = no answer. `ok` = 1 if our grader marks the answer correct. Added 2026-10-18 from the same raw runs; every earlier column is unchanged.
