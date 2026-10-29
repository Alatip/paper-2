# Set A — exploratory data of paper 2

Model answers from our own runs in October 2026 on the item corpus of paper 1 (MMLU-Pro, GPQA Diamond, GSM-Hard and an
olympiad layer of AIME problems). The core is the 236 items of the `tn5-*` files (the first 60 per layer), 227 of them in the
paper-1 corpus after the label audit (58 / 58 / 57 / 54); the paper reports the 227. Set A is exploratory:
the paper reports it, but draws no confirmatory verdicts from it (those come from set B).

Only identifiers, short final answers, correctness, token counts and costs are stored. No prompts, no reasoning, no
question or option text. GPQA Diamond answers are option letters or opaque per-item codes (`X1`, `X2`, … or `~k`): two
answers to the same item share a code if and only if they are the same answer.

## Capped and uncapped files
The runs first capped each answer at 8,192 output tokens. Answers that hit the cap were rerun at 65,536. Where a folder has
both `<name>.jsonl.gz` and `<name>.uncapped.jsonl.gz`, the second one is the first with the cut answers replaced by the
rerun; the paper reads the uncapped files. Replaced rows carry `src` (`rerun2`, `p1rerun`, …; `*-length` = hit 65,536
again, counted wrong). Decomposition is the exception: its protocol fixes the budget at 8,192 tokens, so the paper reads
`runs.jsonl` as is.

The environment variable `P2_DATA=capped` makes the set-A loaders (`operators/p2io.py`) read the capped files instead of
the uncapped ones (default `P2_DATA=uncapped`).

## Common fields
`id` item id (as in paper 1), `layer` (`mmlu-pro`, `gpqa-diamond`, `gsm-hard`, `olympiad`), `model` / `agent`, `answer`
(the final answer as graded), `correct` (true / false / null = cut by the token cap, a missing vote), `finish`,
`tok_in`, `tok_out`, `tok_think` (token counts as returned), `served`, `provider` (who actually served the call).

## Folders

**`tn5-matrix/`** — the solve matrix: one temperature-0 answer per model and item. Models: Claude Haiku 4.5, Claude
Sonnet 5.5, Claude Opus 5.5, DeepSeek V3.2, Mistral Large, Qwen 3.7 Plus, GLM 5.3 Flash. Round 0 of the debate and the pool
for "best single model" and "oracle".
- `rows.jsonl.gz`, `rows.uncapped.jsonl.gz` — one row per (model, item), common fields.
- `matrix-v2.csv` — the same as a 0/1 table: `item_id, dataset`, one column per model (empty = not run).

**`tn5-samples/`** — repeated temperature-1 answers for self-consistency: DeepSeek V3.2 (16 per item), Claude Haiku 4.5
(8 or 16), GLM 5.3 Flash (16). `rows*.jsonl.gz`: common fields plus `sample` (draw number).

**`tn5-debate-mixed/`** — debate of six different models (DeepSeek V3.2, GLM 5.3 Flash, Mistral Large, Qwen 3.7 Plus,
Claude Haiku 4.5, Claude Sonnet 5.5), 3 rounds; each agent sees the others' latest answers and answers again. Round 0 is in
`tn5-matrix`. `rows*.jsonl.gz`: common fields plus `round` (1–3).

**`tn5-debate-homo/`** — debate of six copies of DeepSeek V3.2 (agents `deepseek/deepseek-v3.2#0..5`), 3 rounds; round 0 =
draws 0–5 of `tn5-samples`. Three variants (`arm`): `plain` (sees the other copies), `selfonly` (reconsiders only its own
answer), `pressure` (told to follow the majority). `rows*.jsonl.gz`: common fields plus `arm`, `round`.

**`tn5-aggregate/`** — one DeepSeek V3.2 call reads several answers and writes one. `mode = synth`: the inputs are the six
round-3 answers of the mixed debate; `mode = moa`: five of its own temperature-1 draws. `rows*.jsonl.gz`: common fields plus
`mode`, `inputs` (list of {answer, correct}), `any_input_correct`, `novel` (the output is not among the inputs).

**`debx-weak-pool/`** — the same operators on a pool without frontier models: Claude Haiku 4.5, DeepSeek V3.2, Mistral
Large 2512, GLM 5.3 Flash; 733 items that all seven corpus models answered.

| file | rows | content |
|---|---|---|
| `items.jsonl.gz` | 733 | `id, layer, group`: contested 146 / allwrong 10 / control 40 / allright 537 |
| `round0.jsonl.gz` | 2,932 | first answers (`id, layer, agent, answer, correct`) |
| `debate-w*.jsonl.gz` | 2,352 | debate on 196 items (contested + allwrong + control), `round` 1–3, plus `finish, served, tok_out, cost` |
| `synth-w.jsonl.gz` | 156 | synthesis over round 0 (contested + allwrong): `answer, correct, novel, any_input_correct, cost` |
| `judge-w*.jsonl.gz` | 468 | a judge picks one of the candidate answers (not reported in paper 2) |

**`decomposition/`** — decomposition: the model splits the problem into 2–4 subtasks, solves them in order and assembles
the answer. Bases DeepSeek V3.2 and GLM 5.3 Flash; `arm2` = the base solves the subtasks itself, `arm3` = the subtasks go to
other models (Qwen 3.7 Plus, Mistral Large 2512, Kimi K2.5). 97 items, once each, and three more times on the 61 hard ones.

| file | content |
|---|---|
| `runs.jsonl` | 560 runs: `base, arm, rep, id, dataset, correct, fail, cost` (USD), `tokens, n_subtasks, answered, answer, gold` (not for GPQA), `calls` (per step: finish, served model, output tokens, worker, substituted) |
| `pilot.jsonl` | 186 pilot runs of an earlier protocol, same fields (not pooled) |
| `selection.json` | the 97 items: `[base, layer, id, stratum, solved of 16, cut of 16]` |
| `single-draws*.csv` | the single-model baseline: 2 bases × 3 layers × 60 items × 16 draws — `correct, relabeled, answer` (not for GPQA, ≤ 40 characters), `finish`, tokens, cost |

**Inputs from paper 1 and earlier runs** (each folder has its own `README.md` with source, files and fields):

| folder | what |
|---|---|
| `corpus-mmlu-pro/`, `corpus-gpqa-diamond/`, `corpus-gsm-hard/`, `corpus-olympiad/` | paper 1's item corpus by layer: item ids and where to download each item, golds (GPQA as letters), the uncapped T = 0 answers of seven models |
| `label-audit/` | the two waves of the gold-label audit: which items are excluded, which golds are corrected |
| `tn5-personas/` | DeepSeek V3.2 under four role prompts (diversity by role) |
| `budget-claude-cost/`, `budget-debates/`, `budget-deepseek-cut/`, `budget-forcing/` | earlier runs with token counts and costs, for the equal-budget comparisons |
| `debx-six-debate/`, `debx-ct-debate/` | stored six-agent debates on the corpus, for answer flips and bounds |

## Where it came from
- `tn5-*`: the runs of tech-note 5 (DOI 10.5281/zenodo.23252753).
- `debx-weak-pool`: a preregistered run of 2026-10-12 ($2.86 OpenRouter + Haiku on a subscription).
- `decomposition`: the runs of tech-note 9 (code DOI 10.5281/zenodo.23274139), 9–10 October 2026, $5.73.

The items themselves are paper 1's corpus: `corpus-<layer>/` in this folder lists them and where to download each layer.
