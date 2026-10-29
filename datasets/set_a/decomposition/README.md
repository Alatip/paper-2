# decomposition

Decomposition: the model splits the problem into 2–4 subtasks, solves them in order and assembles the answer. Bases
DeepSeek V3.2 and GLM 5.3 Flash; `arm2` = the base solves the subtasks itself, `arm3` = the subtasks go to other models
(Qwen 3.7 Plus, Mistral Large 2512, Kimi K2.5). 97 items, once each, and three more times on the 61 hard ones. The
protocol fixes the budget at 8,192 tokens, so `runs.jsonl` is read as is.

| file | content |
|---|---|
| `runs.jsonl` | 560 runs: `base, arm, rep, id, dataset, correct, fail, cost` (USD), `tokens, n_subtasks, answered, answer, gold` (not for GPQA), `calls` (per step: finish, served model, output tokens, worker, substituted) |
| `pilot.jsonl` | 186 pilot runs of an earlier protocol, same fields (not pooled) |
| `selection.json` | the 97 items: `[base, layer, id, stratum, solved of 16, cut of 16]` |
| `single-draws.csv`, `single-draws.uncapped.csv` | the single-model baseline: 2 bases × 3 layers × 60 items × 16 draws — `correct, relabeled, answer` (not for GPQA, ≤ 40 characters), `finish`, tokens, cost |

Source: our runs (see "Where it came from" in `../README.md`). Only identifiers, short final answers, correctness, token
counts and costs; GPQA Diamond answers are option letters or opaque per-item codes. License: CC BY 4.0 for our answers and
correctness; the items keep their source terms (see `../../../LICENSE-DATA.md`). Field definitions shared by all set-A folders:
`../README.md`, section "Common fields".
