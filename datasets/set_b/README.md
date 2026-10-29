# Set B — confirmatory experiment of paper 2

200 short-answer items, a pool of 7 models, five generative operators. The analysis plan was fixed before the operator runs.
Item texts and model outputs are not published here (the item sources forbid redistribution); the folder holds which items
were used, where to get them, and whether each answer was right.

## Files

**`items.csv`** — the 200 items and where to download them. One row per item.

| column | meaning |
|---|---|
| `id` | our item id (used in the other files) |
| `source` | benchmark name |
| `hf_dataset`, `hf_config`, `hf_split` | the Hugging Face dataset that holds the item |
| `key_field`, `key_value` | the row in that dataset: the row whose field `key_field` equals `key_value` |
| `area` | mathematics, physics, biology, computer science, other |
| `tier` | `medium` (170) or `hard` (30), see below |
| `flag` | `easy-fill`: 4 physics items added to complete the medium tier although Haiku 4.5 solved them 3/3 |
| `decomp` | position in the 120-item decomposition subset (empty = not in it) |
| `audit` | result of the gold audit, empty = not audited or nothing found (see below) |

**`matrix.csv`** — the solve matrix: every model's first answer on every item. One row per (item, model).
`id, model, draw, correct, finish, tok_out, provider` — `correct` 1/0, `finish` = how the answer ended (`stop`, `length` =
hit the token cap, counted wrong), `tok_out` = output tokens, `provider` = OpenRouter provider that served it (empty for
Claude). DeepSeek V3.2 and GLM 4.7 Flash also have `draw` 2–5 (self-consistency draws).

**`answers.csv`** — every answer the analysis uses, all operators. One row per answer.
`id, op, model, slot, round, correct, finish, tok_out, usd, usd_estimated, code`.
- `op = draw` — single answers; `slot` = draw number. These are the rows of `matrix.csv` plus 400 rows that `matrix.csv`
  does not have: Claude Opus 4.6 draws 2–3 (slots 2 and 3, 200 items each), used only by operator `sc.opus3`.
- `op = copies` — debate between 5 copies of DeepSeek V3.2; `slot` = copy 1–5, `round` 1–2.
- `op = debate7` — debate between the 7 pool models, `round` 1–2.
- `op = synth` — Opus 4.6 reads the 7 first answers and writes its own.
- `op = decomp` — decomposition: the model plans 2–4 subtasks, solves them in order, assembles the answer; one row per chain.
- `sc.opus3` (computed from the `draw` rows, no rows of its own) — self-consistency of Opus 4.6: plurality of draws 1–3
  vs draw 1. Added after the preregistration (a deviation of 29 Oct 2026); its McNemar test is in the Holm family
  of 9 (the 8 preregistered operator tests + this one).
- Debates were run only on items where the first answers were not unanimous; the other items have no debate rows.
- `correct` — 1/0 after the gold audit. `usd` — dollars per call (OpenRouter: billed; Claude: list price, the calls were
  made on a subscription; `usd_estimated = 1`: a Claude call cut at the cap, priced as 64,000 output tokens).
- `code` — answer code `~k`: two answers to the same item have the same code if and only if they are the same answer
  (numbers compared at 12 significant digits, fractions evaluated); empty = no answer. Used for majority votes.

## How the items were chosen
Candidates came from Humanity's Last Exam, MathArena, OlympiadBench (physics, English) and LAB-Bench (SeqQA), short
answers only. Each candidate was tried by Claude Haiku 4.5 up to three times. `medium` = Haiku solved it at least once
(plus 6 ArXivMath items solved by Sonnet 4.6); `hard` = a random sample of items Haiku never solved. Because the medium
tier was picked with Haiku, Haiku's accuracy here is biased upwards.

## Gold audit
The 23 items that all seven models got wrong were reviewed one by one (by Claude Opus 5.5, rechecked by Claude
Fable 5.1): `relabel` — the reference answer was wrong and was corrected (1); `grader_fix` — the right answer was written in a
form the automatic grader could not read (3); `defective` — the item cannot be answered as stated (3; kept in the main
analysis, dropped in a separate sensitivity analysis); `ambiguous_kept` (1).

## Reproducing it yourself
1. Download each item by `hf_dataset` / `hf_config` / `hf_split` and the row with `key_field = key_value`
   (`cais/hle` is gated: request access on Hugging Face first).
2. Ask each model with one fixed prompt. System: `Answer the question. Think briefly if needed, then give the final answer
   on the last line exactly as: FINAL: <answer>`; after the question: `Give the final answer in simplest form on the last
   line as: FINAL: <answer>`. Output cap 64,000 tokens, reasoning effort medium, the provider's default temperature.
3. Models: Claude Haiku 4.5, Sonnet 4.6, Opus 4.6; DeepSeek V3.2, GLM 4.7 Flash (provider Novita), Qwen 3.8-27B, Kimi K2.5
   (OpenRouter).
4. Operators: self-consistency = majority of 5 draws; debates = 2 rounds, each agent sees the others' latest answers and
   the last 1,500 characters of their reasoning, then answers again; synthesis = Opus 4.6 sees the 7 first answers with
   the same excerpts and gives one answer; decomposition = plan (reasoning off, up to 4,096 tokens), each subtask up to
   64,000 tokens, assembly.
5. Grade each answer and compare with `matrix.csv`. Our grader works as follows:
   - an answer that hit the token cap (`finish = length`) is wrong;
   - the answer is the text after the last `FINAL:`; if there is none, the content of the last `\boxed{...}`; if neither, wrong;
   - cleaning: drop `$`, `\boxed{}`, a leading `x =` (keep the right-hand side of the last `=`, `\approx` counts as `=`), a
     trailing unit (`12.5 m/s` -> `12.5`, `\text{...}`); if that does not parse, take the part before "or" / "≈", then a
     leading number with an optional `×10^k`;
   - comparison by item type: integer — must equal the reference exactly (an expression such as `2^{10}` counts); number
     with a tolerance — within the source's relative / absolute tolerance or interval (the answer may be any closed
     form); closed form — numerically equal within relative 1e-6; expression with variables — `simplify(a − g) = 0`
     (SymPy), else equal within 1e-6 at 6 random points of the reference's variables (the answer may not use other
     variables); term or multiple-choice letter — equal after ignoring case, spaces and punctuation.

   The grader code is not in this repository; any implementation of these rules should agree with `correct` except on
   rare answers in unusual formats.

Model outputs are random, so a rerun gives the same picture within the confidence intervals, not the same rows.
Our cost: $529 in total for the preregistered operators (Table B.4 of the paper: OpenRouter as billed, Claude at list price), plus the two extra Claude Opus 4.6 draws per item used only by `sc.opus3` (Section 4.2.1), which Table B.4 does not include.
