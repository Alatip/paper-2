# Data licence

## Part 1. Generic terms (same in every paper repository)

### 1.1 What this file covers

This file covers the data files in this repository (everything under `datasets/`, and any other CSV, JSON or
gzipped table that holds measurements). The code (scripts, library code, tests, READMEs) is covered by the MIT
licence in `LICENSE`, not by this file.

### 1.2 Our own derived data: CC BY 4.0

The following are our own work and are released under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/):

- answers produced by language models in our runs, stored as parsed final answers (a letter, a number or a short
  token), with 0/1 correctness and token counts;
- 0/1 correctness matrices that we computed from third-party outputs (see 1.3 for the limits);
- judge picks and certification verdicts (structured fields only);
- label-audit decisions (item id, decision, old/new gold, reason where the source terms allow it);
- name tables (canonical model names, families, scaffold names).

Attribution: cite the paper (reference in Part 2) and the Zenodo record of the repository.

### 1.3 Third-party sources: upstream terms take precedence

Every task set and every public response matrix we measure on comes from a third party. For each one, the table
in Part 2 states what we store, the upstream licence as we read it on the day given there, and the pinned
revision where one exists. Three rules hold for all of them:

1. **We store derived facts only**: item identifiers (row numbers or upstream ids), gold answers as a letter or a
   short number, model answers as a letter or a short number, 0/1 correctness, model and submission names. We do
   not store question text, answer options, prompts, model reasoning, patches, trajectories or logs. Where public
   text is needed to rerun something, the folder's `get.py` fetches it from the upstream source into a local,
   git-ignored `fetched/` directory (paper 2 has no fetch scripts; `datasets/set_b/items.csv` and the corpus READMEs say
   where to download).
2. **Upstream terms take precedence** over our CC BY 4.0 grant for anything that is still the upstream work.
   Our grant covers our contribution (the measurements and the arrangement); it does not and cannot relicense
   the underlying questions, problems, model outputs or leaderboard data. A reuser who needs the upstream
   material must obtain it under the upstream terms.
3. **Where the upstream states no licence**, we keep only 0/1 correctness (and the names needed to label rows and
   columns), with attribution, and we mark the source as a licence doubt in Part 2. If a rights holder objects,
   the fallback is to remove the stored matrix and ship only the fetch-and-rebuild scripts.

### 1.4 GPQA clause

The GPQA dataset (Rein et al.) is released under CC BY 4.0 with a request from its authors that the questions
not be revealed in plain text online, and access to it is gated on Hugging Face. We follow that request. For
GPQA items this repository stores only: the Hugging Face dataset reference and row number, the deterministic
option-shuffling code, per-model answer **letters**, gold **letters** and 0/1 correctness. No question text,
no option text, no hashed option ids, no model reasoning, and no audit reasons that paraphrase a question.
Anyone who reruns the GPQA layer must accept the dataset terms on Hugging Face themselves; our scripts never
write GPQA text anywhere that is committed.

### 1.5 Humanity's Last Exam

HLE (cais/hle) is released under MIT, gated on Hugging Face, and its authors ask that the questions not leak into
training data. For HLE items this repository stores only the item id, the parsed final answers as **opaque
per-item codes** (`~k`: two answers to the same item share a code if and only if they are the same answer), 0/1
correctness, token counts and costs. No question text, no gold answers in clear, no model reasoning.

### 1.6 Competition problems

Olympiad problems (AIME, HMMT and similar) are the property of their organisers (MAA, HMMT). Copies on Hugging
Face carry the copier's licence, not the organiser's. We never store problem text; we store row numbers, gold
answers (short numeric or LaTeX answers), the models' parsed final answers and correctness.

### 1.7 Takedown and contact

If you hold rights in any upstream material and believe that something stored here goes beyond derived facts, or
should not be redistributed, write to Aziz Latipov (aziz.latypov@gmail.com) or open an issue in the repository.
The item will be removed or replaced by a fetch script.

### 1.8 No warranty

The data are provided as they are, without any warranty of accuracy, completeness or fitness for a purpose. The
label audit documented in the paper corrected the upstream golds it found wrong; it did not check every item.

---

## Part 2. Per-paper table (paper 2)

Paper: Aziz Latipov, "Generative operators in language-model ensembles: what they add to their starting point and what
they cost" (2026), paper 2 of a series. Code and data: https://github.com/Alatip/paper-2, archived at Zenodo, DOI 10.5281/zenodo.23756146 (https://doi.org/10.5281/zenodo.23756146).

Set B (`datasets/set_b/`): what we store for every source is the item id (upstream key), tier and flags, the parsed
final answers of our runs (a letter, a short number or an opaque code `~k`), 0/1 correctness, token counts and costs.
No question text and no model reasoning. Upstream licences read on 2026-10-29 from the Hugging Face dataset cards.

| upstream source | items | upstream licence | revision seen | note |
|---|---|---|---|---|
| cais/hle (HF, gated) | 104 | MIT | `5a81a4c7` | clause 1.5; answers as opaque codes only |
| MathArena/aime_2025, aime_2026, apex-shortlist, apex_2025, brumo_2025, cmimc_2025, hmmt_feb_2025, hmmt_feb_2026, hmmt_nov_2025, smt_2025 (HF) | 45 | CC BY-NC-SA 4.0; problems (c) their organisers | card sha per set | clause 1.6; the NC clause may bind reuse of the problems, not of our numbers |
| MathArena/arxivmath-0626, arxivmath-0826, arxivmath-0926, arxivcs-0826 (HF) | 24 | CC BY-SA 4.0 | card sha per set | — |
| HuggingFaceH4/aime_2024 (HF) | 2 | **none stated**; problems (c) MAA | `2fe88a2f` | clause 1.6 |
| Hothan/OlympiadBench, `OE_TO_physics_en_COMP` (HF) | 17 | Apache-2.0 | `91184b52` | — |
| futurehouse/lab-bench, SeqQA (HF) | 8 | CC BY-SA 4.0 | `5c77cec6` | — |

Set A (`datasets/set_a/`): the paper-1 corpus folders (`corpus-mmlu-pro`, `corpus-gpqa-diamond`, `corpus-gsm-hard`,
`corpus-olympiad`, `label-audit`) carry the same terms as in the paper-1 repository (MMLU-Pro MIT, GPQA Diamond
CC BY 4.0 with clause 1.4, GSM-Hard MIT, olympiad layer as in clause 1.6); each folder's `README.md` repeats them.
All other set-A folders (`budget-*`, `debx-*`, `decomposition`, `tn5-*`) are our own runs on those items: CC BY 4.0
for our answers and correctness, with the item-source terms above; GPQA items there appear only as option letters or opaque per-item codes, or as GOLD/W labels in the `budget-*` folders.

| folder | upstream source | upstream licence | our terms |
|---|---|---|---|
| `model-names`, `model-prices` | our tables and billing records | — | CC BY 4.0 |
