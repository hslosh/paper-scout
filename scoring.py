"""
scoring.py — paper scoring logic, profile-aware.

Scoring (0–9, +bonuses):
    Citations (age-adjusted)    : 0–3 pts
    Recency                     : 0–2 pts
    Abstract quality signals    : profile-dependent (see below)
    Open access / SSRN direct   : +1 pt bonus

Rigor gate: a paper scoring solely on citations + recency is capped at 4
regardless of raw total. To score ≥ 5 it must mention OOS testing OR
transaction costs — this holds across every profile.

Profiles
  general (default)
    +1  out-of-sample / OOS / holdout
    +1  transaction costs / slippage / commissions
    +1  futures/index/ETF (not individual stocks)
    +1  simple / rule-based mechanical strategy
    -1  no empirical test (purely theoretical)
    -1  crypto / NFT / DeFi
    -1  HFT / microsecond / order-book microstructure only   <- PENALTY
    -1  deep learning / neural network without a baseline

  microstructure
    Same as general EXCEPT the HFT/order-book signal flips sign: it's the
    whole point of this profile, so it scores +1 instead of -1.
"""
import re
from datetime import date

# ── Shared signal patterns ───────────────────────────────────────────────────

_OOS_PATTERN   = r"\bout[- ]of[- ]sample\b|holdout|\boos\b"
_COST_PATTERN  = r"transaction cost|slippage|commission|round[- ]trip"
_FUT_PATTERN   = (r"futures (market|contract|trading|return)|equity index|stock index"
                  r"|e-?mini|nasdaq|s&p|\betf\b")
# Anchored to trading context — avoids "Kalman filter", "thresholding wavelets", etc.
_RULES_PATTERN = (r"(?:\bsimple rule\b|\brule[- ]based\b|\bmechanical\s+strateg"
                  r"|\btrading filter\b|\bsignal threshold\b)")
_NOEMP_PATTERN = r"no empirical|purely theoretical|theoretical framework only"
_CRYPTO_PATTERN= r"\bcrypto\b|bitcoin|ethereum|\bdefi\b|\bnft\b"
_HFT_PATTERN   = r"high[- ]frequency|microsecond|nanosecond|order book microstructure"
# Lookahead wraps the FULL alternation (not just "transformer"), bounded to 80 chars
# so a distant "out-of-sample" mention can't rescue a pure-ML paper.
_ML_PATTERN    = (r"(?:deep learning|neural network|lstm|transformer)"
                  r"(?!.{0,80}?(?:simple|baseline|rule|out[- ]of[- ]sample))")

# Microstructure-profile queries deliberately hunt for exactly what the HFT
# penalty screens out in general mode: iceberg orders, hidden liquidity, order
# flow, stop hunts, dealer hedging flows. Their positive signal reuses the
# same pattern the general profile penalizes with.
_MICROSTRUCTURE_PATTERN = _HFT_PATTERN

SCORE_PROFILES = {
    "general": {
        "positive": [
            (_OOS_PATTERN,   1, "OOS test"),
            (_COST_PATTERN,  1, "costs included"),
            (_FUT_PATTERN,   1, "futures/index"),
            (_RULES_PATTERN, 1, "simple rules"),
        ],
        "negative": [
            (_NOEMP_PATTERN,  -1, "no empirical test"),
            (_CRYPTO_PATTERN, -1, "crypto/DeFi"),
            (_HFT_PATTERN,    -1, "HFT only"),
            (_ML_PATTERN,     -1, "ML-only (no baseline)"),
        ],
    },
    "microstructure": {
        "positive": [
            (_OOS_PATTERN,           1, "OOS test"),
            (_COST_PATTERN,          1, "costs included"),
            (_FUT_PATTERN,           1, "futures/index"),
            (_RULES_PATTERN,         1, "simple rules"),
            (_MICROSTRUCTURE_PATTERN, 1, "order-book microstructure"),  # flipped
        ],
        "negative": [
            (_NOEMP_PATTERN,  -1, "no empirical test"),
            (_CRYPTO_PATTERN, -1, "crypto/DeFi"),
            (_ML_PATTERN,     -1, "ML-only (no baseline)"),
            # HFT penalty removed — it's the +1 above in this profile
        ],
    },
}

# Rigor signals — at least one must fire for a paper to score above 4.
# Held constant across profiles: "did they actually test this" matters
# regardless of what kind of edge the paper is chasing.
RIGOR_PATTERNS = [_OOS_PATTERN, _COST_PATTERN]


def score_abstract(abstract: str, year: int, citations: int,
                    oa_pdf: str = "", ssrn_id: str = "",
                    profile: str = "general") -> tuple[int, list[str]]:
    """
    Score a paper. Returns (score, reasons).

    Rigor gate: a paper that never mentions OOS testing or transaction costs is
    capped at 4 regardless of citations + recency. To score >= 5 the abstract
    must contain at least one hard falsification signal.
    """
    if not abstract:
        return 0, ["no abstract available"]
    if profile not in SCORE_PROFILES:
        raise ValueError(f"unknown scoring profile '{profile}' (choices: {list(SCORE_PROFILES)})")

    text  = abstract.lower()
    score = 0
    notes = []

    # Citation score (age-adjusted) — max 3 pts
    current_year = date.today().year
    age = max(1, current_year - (year or current_year))
    cites_per_year = citations / age
    if cites_per_year >= 20:
        score += 3; notes.append("citations: high (>=20/yr)")
    elif cites_per_year >= 5:
        score += 2; notes.append("citations: moderate (5-20/yr)")
    elif cites_per_year >= 1:
        score += 1; notes.append("citations: low (1-5/yr)")

    # Recency — max 2 pts
    if year and year >= current_year - 2:
        score += 2; notes.append("recent (<=2yr)")
    elif year and year >= current_year - 5:
        score += 1; notes.append("recent (<=5yr)")

    # Open access / SSRN direct — +1 pt (an open PDF or SSRN paper means you can read it)
    if oa_pdf or ssrn_id:
        score += 1; notes.append("+1 open access")

    profile_signals = SCORE_PROFILES[profile]
    for pattern, pts, label in profile_signals["positive"]:
        if re.search(pattern, text, re.IGNORECASE):
            score += pts; notes.append(f"+{pts} {label}")

    for pattern, pts, label in profile_signals["negative"]:
        if re.search(pattern, text, re.IGNORECASE):
            score += pts; notes.append(f"{pts} {label}")

    raw_score = max(0, score)

    has_rigor = any(re.search(p, text, re.IGNORECASE) for p in RIGOR_PATTERNS)
    if not has_rigor and raw_score > 4:
        notes.append("capped at 4 (no OOS/costs signal)")
        raw_score = 4

    return raw_score, notes
