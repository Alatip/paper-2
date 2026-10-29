# Datasets of paper 2

Data only: one folder per table, each with a `README.md` (source, license, files, fields). No scripts: the README says
where each source comes from, and the analysis code in `operators/` reads these files directly.

| folder | what |
|---|---|
| `set_b/` | the confirmatory experiment: items (where to download them), solve matrix, answer codes, how to rerun it |
| `set_a/` | the exploratory data: our runs on paper 1's corpus, the corpus itself, its label audit and the earlier runs used for debate and budgets (see `set_a/README.md`) |
| `model-names/` | one canonical name per model, with version, for set A (set B model names are in `set_b/README.md`) |
| `model-prices/` | price per input and output token for every model, for budgets in dollars |

## What is stored

Only derived facts: item ids or dataset row numbers, short final answers, 0/1 correctness, token counts, costs, model
names. No question or option text, prompts or model reasoning. GPQA Diamond and HLE items appear only as identifiers;
their answers are option letters or opaque codes (`X1`, `~k`), equal for equal answers. Answers that a parser left as a
piece of reasoning are replaced by `UNPARSED~<hash>`, where equal strings share a code. Correctness is unaffected.

The license of each source is in its folder's `README.md`. Our own model outputs are released with this repository.
