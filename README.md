# Paper 2: generative operators in LLM ensembles — data and analysis code

Data and code behind paper 2 of the series on LLM ensembles. Paper 2 looks at generative operators, which produce a new
answer instead of picking one: self-consistency, sources of diversity (model family, role, temperature), debate,
synthesis, decomposition, and comparisons at an equal budget in dollars.

Nothing here calls a model. Every script reads the stored tables in `datasets/` and recomputes the paper's numbers
offline, at no cost. Requirements: Python 3.9–3.12 (the pins do not install on 3.13), numpy and scipy (`pytest` for the tests): `python3 -m pip install -r requirements.txt` (pinned to
the versions the checks were run with; tested on Python 3.9.6).

## Two sets of data

- **Set A, exploratory** (`datasets/set_a/`). These are our runs of October 2026 on the item corpus of paper 1: MMLU-Pro,
  GPQA Diamond, GSM-Hard and an olympiad layer. The paper reports set A but draws no confirmatory verdicts from it.
- **Set B, confirmatory** (`datasets/set_b/`). A preregistered experiment: 200 items, 7 models, every operator.
  `datasets/set_b/README.md` describes the items, the files and how to rerun the experiment from scratch.

Set A also holds paper 1's corpus (the items and first answers it starts from), the label audit and the earlier runs used
for the budget and debate sections. Model names and prices are at the top of `datasets/`. See `datasets/README.md`.

## Layout

| path | what |
|---|---|
| `datasets/` | data only: one folder per table, each with a `README.md` (source, license, files, fields) |
| `lib/` | shared statistics (bootstrap, McNemar, answer identity) |
| `operators/` | one script per operator on set A: `self_consistency.py`, `diversity.py`, `debate.py`, `synthesis.py`, `debate_weak.py`, `debate_bounds.py`, `decomposition.py`, `budget.py`, `core_report.py`; `setb_report.py` for set B; loaders in `p2io.py` and `p2parts/` |
| `summaries/build2.py` | runs every set-A operator and writes `summaries/out2/*.json` / `*.md` (under a minute) |
| `summaries/out2/` | the stored outputs the paper's numbers are read from |
| `summaries/*.py` | side-table scripts (core vs rest, debate rounds, self-consistency curve, set-B filters, vote identity) |
| `docs/SETB-SPEC.md` | the analysis specification of set B |
| `docs/PREREG.md` | condensed preregistration of set B |
| `tests/` | `pytest tests`: unit tests, data checks, and regressions against the stored outputs in `tests/expected2/` |

## Reproduce

```
python3 summaries/build2.py                       # set-A operators -> summaries/out2
python3 summaries/core_vs_rest.py                 # table A.2 / core vs rest of the corpus
python3 summaries/debate_rounds.py                # debate: does round 3 add anything
python3 summaries/sc_curve.py                     # self-consistency by number of draws
python3 summaries/set_filter.py                   # set-B filters (also --no-mistral)
python3 operators/setb_report.py                  # set B -> summaries/out2/setB.{json,md}
python3 summaries/vote_identity_setb.py           # set B: answer identity of the votes
SETB_NO_DEFECTIVE=1 python3 operators/setb_report.py   # set B without the 3 defective items
pytest tests
```

Some tables of the paper (section 5, table B.6 and the debate rounds of the weak pool) are read from stored outputs in
`summaries/out2/` (`factor2.json`, `h10_tiers_setb.json`, `debate_phi_rounds.json`) whose scripts are not part of this
repository.

`P2_DATA=capped` makes the set-A loaders read the answers capped at 8,192 output tokens instead of the uncapped
reruns (see `datasets/set_a/README.md`).

## What is not here

No question or option text, no prompts as sent, no model reasoning. GPQA Diamond and HLE items appear only as
identifiers. GPQA answers are option letters or opaque codes, and HLE answers are opaque codes `~k`: two answers to the
same item share a code if and only if they are the same answer. Answers that a parser left as a piece of reasoning
are stored as `UNPARSED~<hash>`: equal strings share a code, so votes are counted the same way. To see the items,
download them from their sources (each `README.md` says where).

## Section references in the code

Docstrings and comments in `operators/` and `tests/` cite sections and tables of the working draft v8 of the paper
("v8 §6.4", "v8 Table 9.1"), the version the operators were ported against. The section numbers of the published paper
differ; the numbers themselves are the same and are checked against the stored outputs.

## Licences

Code under MIT (`LICENSE`). Data: our own runs under CC BY 4.0, third-party items under their upstream terms; see
`LICENSE-DATA.md` and each folder's `README.md`.

## Cite as

Aziz Latipov. *Generative operators in language-model ensembles: what they add to their starting point and what they cost.* 2026.
Code and data: https://github.com/Alatip/paper-2, DOI 10.5281/zenodo.23756146 (https://doi.org/10.5281/zenodo.23756146). See also `CITATION.cff`.
