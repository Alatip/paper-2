# debx-six-debate — the stored six-agent mixed debate, as tech-note-5 sees it (v8 §6.4 contaminated control, §6.4 flips)

Six agents (DeepSeek V3.2, GLM-5.3-flash, Mistral Large 2512, Qwen3.7 Plus, Claude Haiku 4.5, Claude Sonnet 5.5), three synchronous
rounds; round 0 = matrix answers. Built 2026-10-18 from our raw runs (not published), label-audit exclusions applied: the 225 items
complete for all agents in all rounds (`tests/expected2/tn5/tables.md` §B, §B2).

Used by `operators/debate_weak.py` for (a) the W votes inside this debate ("+11 points, 26 fixes, 21 on cut-off problems, 16
round-0 ties") and (b) the flip table of §B2 ("29 percent").

| file | content |
|---|---|
| `debate6.jsonl.gz` | the debate rows, answers cut at the 8,192-token limit counted as cut off |
| `debate6.uncapped.jsonl.gz` | the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun |

License: our outputs. Fields: id, layer, agent, round 0-3, answer, correct (null = cut off at the 8,192-token limit), tok_out.
GPQA answers: letters (TN5 option hashes mapped one-to-one per item); non-answers -> per-item codes `~k`. No model text.
Raw GPQA rows are not published.
