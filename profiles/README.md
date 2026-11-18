# Research profiles

A profile tells the router what you can actually test. It's plain JSON, so
start by copying `example_profile.json` into `profiles/private/` (gitignored)
and editing it. Then set `RESEARCH_PROFILE` in `.env` to point at it.

## Sections

| Key | What it does |
|---|---|
| `review_status` | `draft` or `approved`. Set by `research_profile.py --approve`; don't hand-edit |
| `source_documents` | Name → path of the notes the profile is based on (relative to this folder or absolute). Approval stores their hashes; if they change, the profile goes stale |
| `objectives` | What you're trying to trade. `deployable_horizon: "intraday_flat_by_close"` sends multi-day papers on those instruments to `knowledge_only` |
| `instruments` | Tickers you research. `market` (`equity_index`, `crypto`, `commodities`, ...) is how papers are matched to them |
| `datasets` | What data you own. `fields` use the names in `field_vocabulary`. Only `research_usable: true` datasets count. `serves` lets one dataset cover several instruments. Options datasets set `underlying` and optionally `availability_lag` |
| `not_owned` | Notes on data you don't have (for your reference; routing works from `datasets`) |
| `costs` | Your measured trading costs (for your reference; the flag threshold lives in `screen`) |
| `sample_locks` | Periods you've reserved for out-of-sample testing (for your reference; nothing reads market data) |
| `closed_mechanisms` | Ideas you've already tested. Each needs `keywords` and a `source`. A keyword hit on the same `market` routes a paper to `overlaps_closed_work` |
| `structural_constraints` | Free-text lessons that apply to every idea |
| `routes` | Descriptions of each route |
| `field_vocabulary` | The data field names used in `datasets` |
| `screen` | `min_gross_bps.single_leg` flags papers reporting smaller edges; `power_note` explains the low-frequency flag. Remove the block to turn the flags off |

## Data field names

The router looks for these in paper abstracts and checks them against your
datasets: `ohlcv`, `trades_tick`, `aggressor_side`, `quotes_l1`,
`order_book_l2`, `funding_rate`, `open_interest`, `macro_surprises`,
`options_eod_greeks`, `implied_volatility`, `options_open_interest`,
`options_eod_quotes`, `options_signed_flow`, `options_intraday`.

## Checking a profile

```bash
python research_profile.py            # loads it, prints draft / approved / stale
python research_profile.py --approve  # after you've reviewed it
```
