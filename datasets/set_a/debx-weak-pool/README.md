# debx-weak-pool

The same operators on a pool without frontier models: Claude Haiku 4.5, DeepSeek V3.2, Mistral Large 2512, GLM 5.3
Flash; 733 items that all seven corpus models answered. Preregistered run of 2026-10-12.

| file | rows | content |
|---|---|---|
| `items.jsonl.gz` | 733 | `id, layer, group`: contested 146 / allwrong 10 / control 40 / allright 537 |
| `round0.jsonl.gz` | 2,932 | first answers (`id, layer, agent, answer, correct`) |
| `debate-w*.jsonl.gz` | 2,352 | debate on 196 items (contested + allwrong + control), `round` 1–3, plus `finish, served, tok_out, cost` |
| `synth-w.jsonl.gz` | 156 | synthesis over round 0 (contested + allwrong): `answer, correct, novel, any_input_correct, cost` |
| `judge-w*.jsonl.gz` | 468 | a judge picks one of the candidate answers (not reported in paper 2) |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
