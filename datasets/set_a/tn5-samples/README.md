# tn5-samples

Repeated temperature-1 answers for self-consistency: DeepSeek V3.2 (16 per item), Claude Haiku 4.5 (8 or 16), GLM 5.3
Flash (16).

| file | content |
|---|---|
| `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` | common fields plus `sample` (draw number) |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
