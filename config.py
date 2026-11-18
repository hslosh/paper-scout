"""
config.py — central configuration for paper-scout.

Loads secrets/settings from .env (see .env.example) and holds every constant
used across the project: API endpoints, rate limits,
default query sets (per scoring profile), and output paths.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env in the current working directory if present

ROOT_DIR    = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
PDF_DIR     = RESULTS_DIR / "pdfs"
LOG_DIR     = RESULTS_DIR / "logs"


def _path_setting(name: str, default: Path) -> Path:
    """Read a path from the environment; relative paths resolve against the repo."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


# Research profile used for routing. The bundled example works out of the box;
# point RESEARCH_PROFILE at your own copy (profiles/private/ is gitignored).
PROFILE_PATH = _path_setting("RESEARCH_PROFILE", ROOT_DIR / "profiles" / "example_profile.json")

# Where --save-vault writes markdown notes (Obsidian-compatible). Point
# VAULT_PATH at a folder inside your vault to drop notes straight into it.
VAULT_PATH = _path_setting("VAULT_PATH", RESULTS_DIR / "notes")

# ── Secrets / contact info (.env) ───────────────────────────────────────────
SEMANTIC_SCHOLAR_KEY = os.environ.get("SEMANTIC_SCHOLAR_KEY", "")
OPENALEX_KEY         = os.environ.get("OPENALEX_KEY", "")
UNPAYWALL_EMAIL      = os.environ.get("UNPAYWALL_EMAIL", "")
OLLAMA_MODEL         = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# ── Semantic Scholar ─────────────────────────────────────────────────────────
SS_API_BASE   = "https://api.semanticscholar.org/graph/v1"
SS_RATE_LIMIT = 1.1 if SEMANTIC_SCHOLAR_KEY else 3.0   # unauth: 100 req/5min: be conservative
SS_FIELDS     = ",".join([
    "title", "abstract", "year", "citationCount", "influentialCitationCount",
    "externalIds", "openAccessPdf", "publicationVenue", "authors",
    "publicationDate", "referenceCount",
])

# ── arXiv ────────────────────────────────────────────────────────────────────
ARXIV_API_BASE   = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT = 3.0   # arXiv API TOU: no more than 1 request / 3s
# The search API above refuses Python clients (HTTP 406, Sept 2026); the
# default `arxiv` source searches a local index harvested via OAI-PMH instead.
ARXIV_OAI_BASE = "https://oaipmh.arxiv.org/oai"
ARXIV_OAI_SETS = ["q-fin"]
ARXIV_DB       = ROOT_DIR / "data" / "arxiv_index.sqlite"

# ── OpenAlex ─────────────────────────────────────────────────────────────────
OPENALEX_API_BASE   = "https://api.openalex.org/works"
OPENALEX_RATE_LIMIT = 0.2   # polite pool allows much faster; stay gentle by default

# ── Unpaywall ────────────────────────────────────────────────────────────────
UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_API_URL     = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL    = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT_SEC = 120

# ── Shared search defaults ───────────────────────────────────────────────────
MAX_RESULTS = 100   # results to fetch per query, per source

# Default query sets, keyed by scoring profile.
QUERY_PROFILES = {
    "general": [
        "price pattern trading strategy futures",
        "momentum breakout futures systematic",
        "mean reversion trading rule backtest",
        "opening range breakout equity futures",
        "gap trading strategy out-of-sample",
    ],
    "microstructure": [
        "iceberg order detection",
        "hidden liquidity futures",
        "large trade price impact",
        "order flow imbalance reversal",
        "stop hunting liquidity",
        "dealer gamma hedging price impact",
    ],
    # Options discovery. Search coverage only: scoring stays with --profile, and
    # these are starting queries, not claimed edges. options_eod looks for lagged
    # end-of-day options information that could predict futures returns;
    # options_flow collects flow/dealer papers for reading and data planning.
    "options_eod": [
        "lagged option open interest index futures return predictability",
        "option implied skew next day equity index returns",
        "put call open interest next day futures returns",
        "option implied volatility term structure futures predictability",
    ],
    # Mechanisms that can trade several times a day from tick data with aggressor
    # side and quotes. Starting queries only; closed work still reroutes.
    "intraday_flow": [
        "order flow imbalance predicts short-term futures returns",
        "trade sign autocorrelation intraday price impact futures",
        "order flow toxicity VPIN short-term returns",
        "intraday return predictability high frequency equity index futures transaction costs",
        "short-term reversal after large trades futures market",
        "aggressive order clustering intraday momentum",
    ],
    "options_flow": [
        "options order flow predicts index returns",
        "dealer gamma positioning equity index",
        "intraday options hedging futures price impact",
    ],
}

# `arxiv` = local OAI-harvested index (sources/arxiv_oai.py); `arxiv_api` =
# the live search API, opt-in because its edge refuses Python clients.
ALL_SOURCES     = ["semantic_scholar", "arxiv", "openalex", "arxiv_api"]
DEFAULT_SOURCES = ["semantic_scholar", "arxiv", "openalex"]
