# Text corpus: MMLU-Pro layer (553 items)

## Source

TIGER-Lab/MMLU-Pro on Hugging Face, config `default`, split `test` (12,032 rows), https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro ; revision used: `b189ec765aa7ed75c8acfea42df31fdae71f97be`. Item id `mmlupro:<question_id>` (`source_ref` = question_id). Sampling (the item-building script, seed 0): datasets-server pages at offsets 0, 400, ..., 12000 (length 100), stratified by category, 42 per category, 588 items; options lettered A, B, ... in source order.

## License and redistribution

MIT (dataset card `license: mit`). Redistribution of question text is allowed by the license; we still store ids + answers only, to keep one rule for all layers.

## What paper 1 says (https://github.com/Alatip/paper-1)

Section 4.1 of paper 1 (corpus composition after the label audit).

Appendix B, tables B.1-B.3, row MMLU-Pro: 553 items; best of seven Claude Opus 5.5 0.960, plurality 0.931, oracle 0.998, prize 3.8, 1 all-wrong.

## Files

| file | content |
|---|---|
| `items.csv` | one row per item of the layer that was ever run: `id`, `layer`, `category`, `source` (HF dataset:config:split), `source_ref`, `gold_benchmark` (gold as the public source gives it, after the transformation by the item-building script (not distributed)), `gold` (gold used in the paper, after the label audit), `status` (`corpus` = one of the 941; `excluded-wave1` / `excluded-wave2` = removed by the label audit; `incomplete` = some model has no row, so not in the corpus) |
| `answers.csv` | one row per (item, model): `answer` (the parsed final answer; empty = no answer), `correct` (1/0), `source` (`original`, `original>cap`, `rerun`, `rerun-length`, `rerun-error`, `rerun-ungradable`, with `+relabel2` / `+graderfix2` when the second audit wave changed the grade), `out_tok` |
| `run-order.csv` | one column `id`: the corpus items in the order the original runs used (read by `lib/p1stats/data.py`) |

Model answers are the authors' own outputs; model reasoning texts are not stored here. Models (column `model`):
`claude_haiku`, `claude_opus`, `claude_sonnet55`, `deepseek_deepseek-v3.2`, `mistralai_mistral-large-2512`, `qwen_qwen3.7-plus`,
`z-ai_glm-5.3-flash`; temperature 0, one answer each, up to 65,536 output tokens.

## How the stored files were made

Built 2026-10-18 from our raw runs (not published): the merged uncapped corpus (per model and layer) and our item list
(the item-building script, seed 0; not distributed). The corpus is the items with
all seven models graded, minus the items excluded by the two waves of the label audit (`../label-audit/`).

## Getting the item text

The text is not stored here. Download the source above (at the revision given) and take the rows listed in
`items.csv` (`source_ref`); `gold_benchmark` is the gold as the source gives it, so a download can be checked against it.

