# DeepSeek V3.2 under four role prompts (area tn5)

## Source

Our own runs (tech-note 5, October 2026, OpenRouter / Claude CLI). Built 2026-10-18 from our raw runs (not published); the
state whose stored tables are `tests/expected2/tn5/tables.md`.

## License and redistribution

Our outputs. Stored: ids, short final answers (numbers, expressions, MMLU-Pro letters), correctness, finish reason, token
counts, served model / provider. No prompts, no reasoning, no question or option text. GPQA Diamond: option hashes replaced by
letters / opaque per-item codes; the hashed source files are not published. Cost in dollars is
not in the source rows (only token counts).

## What is in it

T = 1 draws of DeepSeek V3.2 with role prompts engineer / formal / skeptic / teacher (model id deepseek_deepseek-v3.2__persona-<role>), MMLU-Pro, GSM-Hard, olympiad (GPQA Diamond not run). Built from our raw runs (not published).

Files: `rows.jsonl.gz` (22,512 bytes) and `rows.uncapped.jsonl.gz` (the same rows with answers cut at 8,192 output tokens replaced by the 65,536-token rerun), one JSON object per source row, in source-file order. Fields: `model`, `layer`, `id`, `sample`, `answer` (final answer as graded; GPQA: option letter, or opaque per-item code X1, X2, ... where the letter is not recoverable without the option text; identity within an item preserved), `correct` (true / false / null = truncated by the token cap, a missing vote), `finish`, `tok_in`, `tok_out`, `tok_think` (token counts as returned), `served`, `provider`.
Rows of items excluded by wave 1 of the label audit (`../label-audit/`, decision `excluded`) are dropped.

## Used by

v8 §5.8 (table D, role). Loaders: `operators/p2parts/tn5.py`.
