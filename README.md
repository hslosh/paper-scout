# paper-scout

Find academic trading papers worth testing, and skip the ones that aren't.

paper-scout searches Semantic Scholar (which indexes SSRN), arXiv and OpenAlex
for quantitative trading research, scores each paper on how backtestable it
looks, and sorts it against a **research profile**: a JSON file describing
the data you actually have and the ideas you've already tried. A paper that
needs order-book data you don't own, or repeats an idea you already ruled out,
gets told so instead of landing on your reading list.

For the few papers that make it through, it can pull the legal open-access PDF
and have a local LLM build an **evidence card**: what the paper claims, what
data it used, whether it tested out-of-sample and after costs. Every quote on
the card is checked word for word against the PDF text.

Everything runs locally. No paid APIs, no cloud LLM.

```
#2  [4/10]  The Impact of Intraday Momentum on Stock Returns: Evidence from S&P500 and CSI300
     Route: candidate_for_review - observations owned for NQ (named), ES (named); still needs a written test plan
            | low_frequency: at most one trade per day
     Trades: at_most_daily  |  bps reported: -
     Score notes: citations: low (1-5/yr) | recent (<=5yr) | +1 open access | +1 futures/index
```

## Quickstart

Needs **Python 3.10+**.

```bash
git clone https://github.com/hslosh/paper-scout.git
cd paper-scout
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: Copy-Item .env.example .env
```

Run a search (Semantic Scholar and OpenAlex need no setup):

```bash
python paper_scout.py "intraday momentum equity index futures" --sources semantic_scholar,openalex
```

Results print to the terminal and save under `results/runs/<run-id>/`.

To include arXiv, build the local arXiv index once (about 10 minutes, then
incremental). After that, the default sources cover all three:

```bash
python paper_scout.py "intraday momentum" --sync-arxiv
python paper_scout.py "order flow imbalance futures"
```

Run the tests (offline, about a second):

```bash
python -m unittest discover -s tests
```

## How it works

1. **Search.** Each query goes to every selected source. Records are merged by
   DOI, arXiv ID, or an exact title + year + author match. Conflicts are kept
   apart instead of guessed.
2. **Score (0–9+).** Citations (age-adjusted), recency, and abstract keywords:
   out-of-sample testing, transaction costs, a liquid instrument, a simple
   mechanical rule. Pure theory, unbaselined deep learning and off-topic
   markets lose points. A paper can't score above 4 on citations and recency
   alone; it has to mention out-of-sample testing or costs. See `scoring.py`.
3. **Route.** Each paper gets one route, with reasons, based on your profile:

   | Route | Meaning |
   |---|---|
   | `candidate_for_review` | You have the data it needs, and it doesn't repeat closed work |
   | `knowledge_only` | Worth knowing, but no tradable claim (or the wrong horizon) |
   | `requires_data` | Needs data you don't have; the missing pieces are listed |
   | `overlaps_closed_work` | Looks like something you already tested |
   | `insufficient_evidence` | No usable abstract, or unclear timing |
   | `irrelevant` | Not finance |

   Routing also tags how often a mechanism can trade (`multi_per_day`,
   `at_most_daily`) and flags reported edges smaller than your cost bar.
   Options papers are also sorted into a separate options research list.
4. **Evidence cards** (`--evidence-cards`). For the top candidates it fetches
   legal open-access full text, picks the pages most likely to hold data,
   method and results, and asks a local Ollama model for a schema-constrained
   summary. Any quote that isn't found verbatim in the text downgrades its
   claim to "uncertain."

Routing and scoring are regex-based triage. They tell you what to read first,
not whether something works.

## Research profiles

The bundled `profiles/example_profile.json` describes a made-up researcher
working on NQ/ES futures with a crypto side project, so everything runs out
of the box. To use your own:

```bash
mkdir -p profiles/private
cp profiles/example_profile.json profiles/private/my_profile.json
# edit it, then in .env:
#   RESEARCH_PROFILE=profiles/private/my_profile.json
python research_profile.py            # check it loads, show status
python research_profile.py --approve  # once you've reviewed it
```

`profiles/private/` is gitignored. See [`profiles/README.md`](profiles/README.md)
for what each section does.

Until a profile is approved, every route is marked provisional. Approval
stores a hash of the profile and of each source document it lists, so editing
either later marks it stale until you review it again.

## Configuration (`.env`)

All optional.

| Variable | What it's for |
|---|---|
| `SEMANTIC_SCHOLAR_KEY` | Free [API key](https://www.semanticscholar.org/product/api); raises the rate limit |
| `OPENALEX_KEY` | Optional OpenAlex key |
| `UNPAYWALL_EMAIL` | Contact email for [Unpaywall](https://unpaywall.org/products/api) PDF lookups (no signup) |
| `OLLAMA_MODEL` | Local model for evidence cards and drafts. Default `qwen2.5:7b-instruct` |
| `RESEARCH_PROFILE` | Path to your profile. Default `profiles/example_profile.json` |
| `VAULT_PATH` | Folder for `--save-vault` notes. Default `results/notes`; point it into an Obsidian vault if you use one |

For evidence cards, install [Ollama](https://ollama.com), then
`ollama pull qwen2.5:7b-instruct`, and make sure it's running.

## Usage

```bash
# Built-in query set (general scoring)
python paper_scout.py

# Microstructure scoring: order-book/HFT papers get a bonus instead of a penalty
python paper_scout.py --profile microstructure

# Other built-in query sets: intraday_flow, options_eod, options_flow
python paper_scout.py --query-profile intraday_flow

# Evidence cards for the top 3 candidates
python paper_scout.py "intraday momentum equity index futures" --evidence-cards

# Evidence cards from the options list instead
python paper_scout.py --query-profile options_eod --evidence-cards --cards-from options

# Ignore routing; keep anything scoring 6+
python paper_scout.py --select score --min-score 6 --top 25

# Write markdown notes for the top 3 kept papers
python paper_scout.py --save-vault
```

| Flag | Default | What it does |
|---|---|---|
| `query` | built-in set | A single search string |
| `--sources` | `semantic_scholar,arxiv,openalex` | Comma-separated. `arxiv` is the local index; `arxiv_api` is the live API (currently rejects most Python clients) |
| `--limit` | `100` | Results per query per source |
| `--profile` | `general` | Scoring mode: `general` or `microstructure` |
| `--query-profile` | same as `--profile` | Built-in query set: `general`, `microstructure`, `intraday_flow`, `options_eod`, `options_flow` |
| `--select` | `route` | Keep rule: `route` keeps `candidate_for_review`; `score` keeps score ≥ `--min-score` |
| `--min-score` | `5` | Score cutoff for display and `--select score` |
| `--top` | `15` | Results shown in the terminal |
| `--sync-arxiv` | off | Update the local arXiv q-fin index before searching |
| `--fetch-fulltext` | off | Download legal open-access PDFs for all kept papers |
| `--evidence-cards` | off | Build verified evidence cards for the top `--draft-limit` papers |
| `--cards-from` | `kept` | Card the keep-list or the `options` list |
| `--draft-hypothesis` | off | Older free-form LLM draft (evidence cards are preferred) |
| `--draft-limit` | `3` | How many papers to card or draft |
| `--save-vault` | off | Write markdown notes for the top 3 kept papers |

Exit codes: `0` complete, `1` every search failed, `2` partial (a source
failed, or an optional step like PDF download failed).

## Output

Every run gets its own folder, so nothing is overwritten:

```
results/runs/<timestamp>_<id>/
├── <query>_all.csv          every paper, merged across sources
├── <query>_kept.csv         the keep-list
├── options_research.csv     options papers with compatibility notes
├── papers.json              full records, provenance, routes
├── raw_records.json         what each source returned
├── manifest.json            arguments, code hashes, per-source status
└── cards/                   evidence cards (.json and .md)
results/pdfs/                cached open-access PDFs
results/notes/               --save-vault notes (or VAULT_PATH)
```

Notes are create-only, so re-running never overwrites your edits.

## Scheduling

`run_scout.sh` (cron) and `run_scout.ps1` (Windows Task Scheduler) run a
weekly scout with evidence cards and log to `results/logs/`. Edit the flags
inside to taste.

```bash
# crontab -e
0 6 * * 1 /path/to/paper-scout/run_scout.sh
```

```powershell
schtasks /create /tn "PaperScout" /sc weekly /d MON /st 06:00 /tr `
  "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"C:\path\to\paper-scout\run_scout.ps1`""
```

## Measuring the router

`evaluate.py` checks routing against your own judgment:

```bash
python evaluate.py replay              # re-route archived runs, compare with old keep-lists
python evaluate.py make-labels --n 60  # blind labeling sheet (routes hidden)
python evaluate.py score results/labels.csv
```

## Project layout

```
paper_scout.py        entry point
config.py             settings, endpoints, built-in query sets
scoring.py            0–9 score
routing.py            routes, feasibility, trade-frequency screen, options list
research_profile.py   load / check / approve a profile
evidence.py           evidence cards with quote verification
identity.py           cross-source record merging
fulltext.py           open-access PDF lookup and text extraction
llm_draft.py          free-form Ollama drafts
vault.py              markdown notes
output.py             terminal output and CSVs
evaluate.py           routing evaluation against human labels
sources/              Semantic Scholar, OpenAlex, arXiv (local OAI index + live API)
profiles/             example research profile
tests/                offline test suite
```

## Notes

- arXiv has no citation counts, so arXiv papers score on recency and content only.
- Full text is only ever fetched from open-access locations the provider or
  Unpaywall declares. Nothing gets around a paywall.
- LLM output is a reading aid. Treat it like a research assistant's first
  pass, not a verdict.
- Not investment advice.

## License

[MIT](LICENSE)
