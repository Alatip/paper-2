# Canonical model names (one table for paper, code and data)

Author's rule (2026-10-17): the paper text, the code and the data use the same model names, with version numbers. `names.csv` is
the only place where a name is defined; everything else reads it.

## Files

| file | content |
|---|---|
| `names.csv` | scope, raw id or string, canonical name, vendor, api_model, legacy_label, source of the version, note |

Scopes (the first column of `names.csv`):

- `corpus`: the seven models of the text corpus, keyed by the corpus id used in `set_a/corpus-*/answers.csv` (`claude_haiku`, ...).
- `corpus-judge`: the same seven models under the runner id (`claude:haiku`, `deepseek/deepseek-v3.2`, ...), the form in which
  the runner records the model it called.
- `budget-debate`: the six copies of DeepSeek V3.2 in the six-copies debate of `set_a/budget-debates` (`agent` = model id `#k`).
- `debx-homo`: the six copies of DeepSeek V3.2 in the homogeneous debate (`deepseek/deepseek-v3.2#k`, k = 0..5).
- `decomp`: the two base models of `set_a/decomposition` (`runs.jsonl` field `base`).
- `decomp-worker`: the three workers of the team arm of `set_a/decomposition` (`runs.jsonl` field `calls[].worker`).
- `tn5-persona`: DeepSeek V3.2 with one of four role system prompts (`set_a/tn5-personas`); the same model as `deepseek_deepseek-v3.2`.
- `tn5-homo-debate`: the six copies of DeepSeek V3.2 in the homogeneous debate of `set_a/tn5-debate-homo`.

## The seven corpus models

| corpus id | canonical name | model actually called | where the version comes from |
|---|---|---|---|
| claude_haiku | Claude Haiku 4.5 | claude-haiku-4-5-20251001 | runner alias table (`haiku`), Claude CLI `--model` |
| claude_opus | Claude Opus 5.5 | claude-opus-5-5 | runner alias table (`opus`), Claude CLI `--model` |
| claude_sonnet55 | Claude Sonnet 5.5 | claude-sonnet-5-5 | runner alias table (`sonnet55`), Claude CLI `--model` |
| deepseek_deepseek-v3.2 | DeepSeek V3.2 | deepseek/deepseek-v3.2 | OpenRouter id, echoed as the served model in every raw row |
| mistralai_mistral-large-2512 | Mistral Large 2512 | mistralai/mistral-large-2512 | same |
| qwen_qwen3.7-plus | Qwen3.7 Plus | qwen/qwen3.7-plus | same |
| z-ai_glm-5.3-flash | GLM-5.3-Flash | z-ai/glm-5.3-flash | same |

For the three Claude models the raw rows record only the alias (`claude:haiku`, ...). The runner's alias table maps it to a full
model id, and that table has not changed since the runner's first commit (2026-09-29), before any corpus run. The CLI response
does not echo the model id, so the version rests on that table, not on a server-side record. The answers and judge runs of
2026-10-05..16 (including the uncapped re-run and the judge runs) used the same aliases.

The canonical names of the OpenRouter models are their OpenRouter ids written as names. No release beyond the id is claimed:
`mistral-large-2512` is written "Mistral Large 2512", not under a marketing name.
