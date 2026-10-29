# Text corpus: olympiad layer, AIME 2024/2025 + HMMT Feb 2025 (83 items)

## Source

- `aime24:<k>` = row k of HuggingFaceH4/aime_2024 (`default`/`train`, 30 rows), https://huggingface.co/datasets/HuggingFaceH4/aime_2024
- `aime25:<k>` = row k of MathArena/aime_2025 (`default`/`train`, 30 rows), https://huggingface.co/datasets/MathArena/aime_2025
- `hmmt25:<k>` = row k of MathArena/hmmt_feb_2025 (`default`/`train`, 30 rows), https://huggingface.co/datasets/MathArena/hmmt_feb_2025

All 90 problems were run; 83 are in the corpus (7 `incomplete`: some model has no graded row).

## License and redistribution

- **Upstream licenses:** MathArena/aime_2025 and MathArena/hmmt_feb_2025: CC BY-NC-SA 4.0 (dataset cards);
  HuggingFaceH4/aime_2024: **no license** on the card. The problems are competition material, copyright MAA (AIME) and HMMT.
- **What we store:** item ids (`aime24:<k>` etc. = HF row numbers), gold answers (integers), the models' parsed final answers
  (numbers), 0/1 correctness and output token counts. **No problem text** and no model reasoning.
- **Position:** problem text is never redistributed; it is fetched from the upstream datasets (links above). Short numeric golds and our
  own outputs are facts, not the problems' expression. MathArena's NC clause may matter for commercial reuse.

## What paper 1 says (https://github.com/Alatip/paper-1)

Section 4.1 of paper 1 (corpus composition after the label audit).

Appendix B, tables B.1-B.3, row olympiad: 83 items; best of seven Claude Opus 5.5 0.964, plurality 0.964, oracle 1.000, prize 3.6.

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

