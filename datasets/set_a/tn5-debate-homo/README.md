# tn5-debate-homo

Debate of six copies of DeepSeek V3.2 (agents `deepseek/deepseek-v3.2#0..5`), 3 rounds; round 0 = draws 0–5 of
`../tn5-samples`. Three variants (`arm`): `plain` (sees the other copies), `selfonly` (reconsiders only its own answer),
`pressure` (told to follow the majority).

| file | content |
|---|---|
| `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` | common fields plus `arm`, `round` |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
