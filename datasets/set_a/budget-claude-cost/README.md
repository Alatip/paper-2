# budget-claude-cost

Claude rows with nominal API cost: Haiku 4.5 T = 1 draws ('samples', 7–16 per item) and one answer each of Haiku 4.5, Sonnet 5.5, Opus 5.5 ('single'). Costs are the CLI's nominal per-call prices recorded in the raw rows (subscription runs); think_tok = reasoning tokens.

- Built 2026-10-18 from our raw runs (not published); state of 2026-10-09.
- License / redistribution: our own model outputs (derived fields only). Item ids refer to MMLU-Pro (MIT), GPQA Diamond (CC BY 4.0, 'do not reveal online': only ids, GOLD/W labels and option letters (`akey`) here; no texts or hashes), GSM-Hard (MIT), AIME/HMMT (see corpus-olympiad).
- Files: rows.csv — model, layer, stage, id, sample, correct, cost_usd, out_tok, think_tok, vote. 6,313 rows (Haiku samples 3,252; singles Opus 1,025, Sonnet 1,018, Haiku 1,018; single rows are not filtered by the label-audit exclusions; the analysis intersects them with the Haiku sample ids).
- `vote`: GOLD = graded correct by our grader (audited gold, 2026-10-08); W1, W2, ... = distinct wrong answers of that item (per item, order of first appearance); empty = no parsable answer. No answer text or letters are stored.
- Used in: v8 §10.5, table 10.2 (cost-matched Q4). Operator: `operators/budget.py` via `operators/p2parts/budget.py`.


## Answer keys (added 2026-10-18)

`akey` = answer key for paper 1's answer-identity rule (operators/p2parts/answer_id.py): MMLU-Pro and GPQA rows hold the option letter our grader parses, or an opaque per-item code `~k` when no letter parses; other layers hold the parsed answer string. Empty = no answer. Added 2026-10-18 from the same raw runs; every earlier column is unchanged.
