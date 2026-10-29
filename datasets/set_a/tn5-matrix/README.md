# tn5-matrix

Solve matrix of set A: one temperature-0 answer per model and item for the seven models (Claude Haiku 4.5, Claude Sonnet 5.5,
Claude Opus 5.5, DeepSeek V3.2, Mistral Large, Qwen 3.7 Plus, GLM 5.3 Flash). Round 0 of the debate and the pool for
"best single model" and "oracle".

| file | content |
|---|---|
| `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` | one row per (model, item), common fields |
| `matrix-v2.csv` | the same as a 0/1 table: `item_id, dataset`, one column per model (empty = not run) |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
