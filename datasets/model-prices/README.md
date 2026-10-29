# model-prices

Price per input and per output token for every model of the paper-2 pool (decided 2026-10-19: budget comparisons in dollars,
never at equal tokens). Cost of a call c = pi_in * t_in + pi_out * t_out (reasoning tokens are output tokens). Used by
`operators/budget.py` (dollar mode, default) through `operators/p2parts/budget.py:prices()`.

- Built 2026-10-19 from the raw call rows of our paper-2 runs and of the 2026-10-18 uncapped rerun (not published; the
  decomp-65k run excluded). Only sums and fitted prices are stored: no ids, no texts.
- `prices.csv`: one row per (model, provider) and one `provider = ALL` row per model (the price the operators use).
  Columns: model, provider, basis, n_calls, n_fit (rows kept after the 50 % residual trim), in_tok, out_tok, cost_usd (sum billed),
  pi_in_per_M, pi_out_per_M ($ per million tokens), fit_rms_usd (per-call rms residual of the fit).
- `basis`:
  - `billed` — OpenRouter's reported cost per call, non-negative least squares on (t_in, t_out) per provider; t_out = completion tokens,
    which already include reasoning tokens.
  - `billed-mix` (ALL rows of OpenRouter models) — per-provider prices weighted by that provider's share of input (resp. output) tokens,
    providers with >= 30 calls; i.e. what OpenRouter's routing actually cost on these runs.
  - `list` (ALL rows of Claude models) — **Anthropic list API price, not billed**: the Claude models ran on the subscription.
    Haiku 4.5 $1 / $5, Sonnet 5.5 $2 / $10, Opus 5.5 $4 / $20 per million (claude-api reference, cached 2026-10-06).
    No cache discount is applied, although the CLI's t_in includes cached system-prompt tokens (~3.5k per call), so list cost overstates
    what an API caller with caching would pay on the input side.
  - `cli-notional` — the Claude CLI's own notional cost per call fitted the same way (it prices cache reads cheaply and, for Opus/Sonnet,
    is closer to $3 / $24 per million); reported for comparison, not used.
- Spread across providers (>= 30 calls): DeepSeek V3.2 input 0.19–3.50 $/M, output 0.31–2.53 $/M (SambaNova is the outlier; without it
  0.19–0.35 / 0.31–1.50); GLM 5.3 Flash input 0.00–0.46, output 0.39–1.62 (Parasail's input fits at 0, Reka is 3x the others); Qwen 3.7
  Plus and Mistral Large 2512 have one provider each.
- License: our own billing records, derived sums only.
