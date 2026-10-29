# debx-ct-debate — live corpus-traces rows used by the axis-H bounds (v8 §6.6, §6.7)

Built 2026-10-18 from our raw runs (not published): the live corpus, NOT label-audited, answers compared as raw strings, exactly as
the bounds analysis (H3, H6c) read it.

| file | rows | content |
|---|---|---|
| `mixed.jsonl.gz` | 5,536 | six-agent mixed debate, 231 ids (229 complete in all rounds = H3 item set); id, layer, agent, round, answer, correct |
| `homo.jsonl.gz` | 12,888 | six DeepSeek V3.2 copies, arms plain / selfonly / pressure; arm, id, layer, agent (`deepseek/deepseek-v3.2#k`), round, answer, correct |
| `deepseek-samples16.jsonl.gz` | 3,568 | DeepSeek V3.2 T = 1 samples, the 223 items with all 16 samples; id, layer, sample, answer, correct; item order = source file order (the H6c bootstrap depends on it) |

Each file has a twin `*.uncapped.jsonl.gz` (`mixed`, `homo`, `deepseek-samples16`): the same rows with answers cut at 8,192 output
tokens replaced by the 65,536-token rerun.

License: our outputs. GPQA: letters only (the live corpus stores letters); non-answers -> per-item codes `~k` (raw-string identity
kept). No model text. Raw GPQA rows are not published. Used by `operators/debate_bounds.py`.
