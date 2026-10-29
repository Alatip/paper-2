# Text corpus: GPQA Diamond layer (164 items)

## Source

Idavidrein/gpqa on Hugging Face (gated: accept the terms, then an HF token), config `gpqa_diamond`, split `train` (198 rows), https://huggingface.co/datasets/Idavidrein/gpqa (revision 83022cef seen 2026-10-16). Item id `gpqa:<k>` = row k of that split in datasets-server order (`source_ref` = k). All 198 rows were run; 164 are in the corpus (33 `incomplete`, 1 excluded).

### Option order (from the item-building script, verbatim)

```python
opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
rng2 = random.Random(k); order = list(range(4)); rng2.shuffle(order)
letters = "ABCD"; gold = letters[order.index(0)]
optxt = "\n".join(f"{letters[i]}. {opts[o]}" for i, o in enumerate(order))
```

The shuffle is **deterministic**: it is seeded per row with `random.Random(k)` and does not depend on any earlier random state. Recomputing it reproduces the stored gold letter for 197/198 rows; the 198th, gpqa:17, is the one item whose gold the label audit corrected (A -> C). The same code, run on 2026-10-16, rebuilt all 198 prompts byte-identical to the ones the models saw. No permutation table is therefore needed.

### Letters instead of hashes

An earlier copy of the corpus (tech note 5) stored GPQA answers and golds as hashed option ids (`g` + 8 hex). The merged corpus maps them back to letters per (model, item) from the live corpus-traces copy; this folder contains letters only (0 hashes; checked). Six GLM-5.3-Flash answers that the parser could not reduce to a letter (graded wrong) are stored as `UNPARSED` instead of the parser output, which contained reasoning text; empty answers stay empty.

## License and redistribution

CC BY 4.0, but the authors ask that the questions not be revealed in plain text online (canary string in the data; gated on HF). **Stored here: no question text, no option text, no model reasoning, no hashed option ids** — only the HF reference + row number, the shuffling code above, per model answer letters, gold letters and correctness. Audit reasons for GPQA items are withheld. Letter answers alone do not reveal the questions.

## What paper 1 says (https://github.com/Alatip/paper-1)

Section 4.1 of paper 1 (corpus composition after the label audit).

Appendix B, tables B.1-B.3, row GPQA Diamond: 164 items; best of seven Claude Opus 5.5 0.921, plurality 0.878, oracle 0.988, prize 6.7, 2 all-wrong (gpqa:32, gpqa:76).

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

