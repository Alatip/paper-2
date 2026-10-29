# tn5-aggregate

One DeepSeek V3.2 call reads several answers and writes one. `mode = synth`: the inputs are the six round-3 answers of
the mixed debate; `mode = moa`: five of its own temperature-1 draws.

| file | content |
|---|---|
| `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` | common fields plus `mode`, `inputs` (list of {answer, correct}), `any_input_correct`, `novel` (the output is not among the inputs) |

`*.uncapped.*` = the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun (the files the
paper reads; `P2_DATA=capped` switches to the capped ones).

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
