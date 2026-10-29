# tn5-debate-mixed

Debate of six different models (DeepSeek V3.2, GLM 5.3 Flash, Mistral Large, Qwen 3.7 Plus, Claude Haiku 4.5, Claude
Sonnet 5.5), 3 rounds; each agent sees the others' latest answers and answers again. Round 0 is in `../tn5-matrix`.

| file | content |
|---|---|
| `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` | common fields plus `round` (1–3) |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
