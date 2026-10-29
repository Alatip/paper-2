# Text corpus: GSM-Hard layer (141 items)

## Source

reasoning-machines/gsm-hard on Hugging Face, config `default`, split `train` (1,319 rows), https://huggingface.co/datasets/reasoning-machines/gsm-hard (HF revision 960448f7 seen 2026-10-16; the Hugging Face datasets-server rows API serves the current revision). Sampling (the item-building script, seed 0): pages at offsets 0, 400, 800, 1200 (length 100), shuffled, first 150.

**Item ids are not reproducible from code.** The item-building script named items `gsmhard:<abs(hash(question)) % 10**8>`; Python's `hash()` of a string is salted per process (PYTHONHASHSEED), so a new run gives other ids. The ids are kept as they are in the paper and in `../label-audit/`; `source_ref` holds the row number in the split (matched by exact question text, 150/150 unique). Gold = `str(target)` (float formatting, e.g. `1103589.0`); grading is numeric.

## License and redistribution

MIT (dataset card). Question text could be redistributed; stored here as ids + row numbers only.

## What paper 1 says (https://github.com/Alatip/paper-1)

Section 4.1 of paper 1 (corpus composition after the label audit).

Appendix B, tables B.1-B.3, row GSM-Hard: 141 items; best of seven Claude Sonnet 5.5 0.986, plurality 0.986, oracle 1.000, prize 1.4.
Appendix A: wave 1 corrected 11 GSM-Hard golds and excluded 6; wave 2 corrected 2 and excluded 3.

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

