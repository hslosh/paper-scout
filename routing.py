"""
routing.py — first-pass feasibility routing against the research profile.

Deterministic and conservative: it reads only title + abstract, so every
route is a triage suggestion for human review, never a verdict. It runs
beside scoring.py (whose score is kept as a within-route tiebreak) rather
than replacing it.

Precedence: insufficient_evidence -> irrelevant -> overlaps_closed_work ->
requires_data -> knowledge_only -> candidate_for_review.
"""
import re

# Finance anchors: a paper needs two distinct ones to count as in-domain, so
# a lone "price" or "futures" (carbon prices, wind-power futures pricing
# theory, iceberg geology) does not qualify on its own.
_FINANCE = {
    "equity": r"\bstock (?:markets?|returns?|prices?|exchanges?|index)|\bstocks\b|\bequit(?:y|ies)\b",
    "futures": r"\bfutures\b|\be-?mini\b",
    "options": r"\boptions?\b(?! (?:for|to|are|is))|implied vol",
    "trading": r"\btrading\b|\btraders?\b",
    "returns": r"(?:stock|asset|excess|abnormal|index|futures|portfolio|daily|intraday) returns?\b|\breturn predictab",
    "volatility": r"\bvolatility\b",
    "liquidity": r"\bliquidity\b|\bbid-ask\b|\bspreads?\b",
    "microstructure": r"order (?:book|flow)|microstructure|market mak",
    "portfolio": r"\bportfolios?\b|\binvestors?\b|\basset pric",
    "index": r"s&p|nasdaq|stock index|equity index|\betfs?\b",
    "crypto": r"\bbitcoin\b|\bcrypto",
    "arbitrage": r"\barbitrage\b|\bhedg(?:e|ing)\b",
    "anomaly": (r"(?:price|intraday|time[- ]series|cross[- ]sectional|return) (?:momentum|reversal)"
                r"|momentum (?:effect|strateg|factor|prediction)|mean[- ]reversion"),
    "evaluation": (r"backtest|overfitting|efficient markets?|market efficiency|sharpe ratio|drawdown"
                   r"|risk premi|regime[- ]switching|downside risk"),
}

_EQUITY_INDEX = r"s&p|nasdaq|e-?mini|stock index|equity index|\bspy\b|\bqqq\b|\bspx\b|\bndx\b|\bvix\b"
_CRYPTO = r"\bbitcoin\b|\bbtc\b|\bcrypto|\bperpetual"
_OTHER_MARKETS = {
    "commodities": (r"crude oil|\bgold\b|\bsilver\b|commodit|natural gas|agricultur|\bcopper\b|electricity|wind power"
                    r"|\bcorn\b|soybean|\bwheat\b|\bhogs?\b|cattle|\bsugar\b|coffee|cotton|\biron ore\b"),
    "fx": r"foreign exchange|\bcurrenc|\bfx\b|exchange rates?",
    "rates": r"treasury|bond futures|interest rate futures|\bbunds?\b|yield curve",
    "single_stocks": r"cross[- ]section|individual stocks|stock selection|long-short portfolio|anomal(?:y|ies)",
}

# Observation types a paper may require -> whether any owned dataset has them
# for a compatible instrument (see profile datasets / not_owned).
# Observation requirements, named with the profile's field vocabulary.
_REQUIRES = {
    "order_book_l2": r"limit order book|order book depth|\blob\b|level[- ]?(?:2|ii)\b|market by order|\bmbo\b|queue (?:position|imbalance)|depth of (?:the )?book",
    "trades_tick": r"tick[- ]by[- ]tick|trade[- ]by[- ]trade|transaction[- ]level|aggressor|signed (?:trades|volume)|order flow imbalance|trade classification",
    "aggressor_side": r"aggressor|buyer[- ]initiated|seller[- ]initiated|signed (?:trades|volume)|order flow imbalance|trade classification",
    "macro_surprises": r"announcement surprise|consensus forecast|surprise component|news sentiment|twitter|social media|text(?:ual)? analysis",
    "funding_rate": r"funding rates?",
}

_INTRADAY = r"intraday|intra-day|high[- ]frequency|\bminutes?\b|\btick\b|open(?:ing)? (?:auction|range)|\bclose\b|last half[- ]hour|\bsession\b"
_MULTIDAY = r"monthly|weekly|holding periods? of (?:one|two|three|several|\d+) (?:weeks?|months?)|rebalanc|buy[- ]and[- ]hold|annual|quarterly"
_THEORY = r"no empirical|purely theoretical|theoretical (?:model|framework)|we (?:derive|prove)|closed[- ]form|stochastic control|equilibrium model"
_PRICING = r"option pricing|pricing (?:of )?options|pricing model|valuation of|delta hedging|hedging strateg(?:y|ies) for options"
# A candidate must claim something tradable, not only describe a market.
_SIGNAL = (r"predict|forecast|strateg|trading rule|\bsignals?\b|reversal|momentum|\bdrift\b|profitab|\balpha\b"
           r"|backtest|excess returns?|abnormal returns?|mean[- ]revers|autocorrelat|seasonalit|anomal")
# Options research lane: options papers stay visible for research even when
# they aren't a trade candidate. Compatibility is judged against the profile's
# options datasets (by underlying).
_OPTIONS_LANE = {
    "flow_positioning": (r"option(?:s)? (?:order )?flow|options? volume|open interest|put[- ]call|dealer|market makers?' "
                         r"(?:inventory|positioning)|gamma exposure|\bgex\b|hedging (?:pressure|demand|flows?)|delta[- ]hedg"
                         r"|net (?:gamma|delta)|0dte|zero[- ]days? to expir|pin(?:ning)? risk|max pain"),
    "implied_information": (r"implied volatility|option[- ]implied|volatility (?:surface|skew|smile)|\bskew\b"
                            r"|variance risk premium|\bvix\b|term structure of (?:implied )?volatility"),
    "pricing_theory": r"option pricing|pricing (?:of )?options|valuation of options|black[- ]scholes",
}
_OPTIONS_ANY = r"\boptions?\b(?! (?:for|to|are|is))|implied vol|\bvix\b|put[- ]call"
_EMPIRICAL = r"\bwe (?:find|document|test|show empirically|estimate)|\bempirical|\bdata\b|\bsample\b|backtest"

_OOS = r"out[- ]of[- ]sample|holdout|\boos\b|walk[- ]forward"
_OOS_NEG = r"(?:no|without|not|lack of|absence of)\s+(?:\w+\s+){0,3}(?:out[- ]of[- ]sample|holdout|walk[- ]forward)"
_COST = r"transaction costs?|slippage|commissions?|round[- ]trip"
_COST_NEG = (r"(?:ignor\w*|without|excluding|before|abstract\w* from|no|zero|neglect\w*)\s+(?:\w+\s+){0,2}(?:transaction )?costs?"
             r"|costs? (?:are|were|is|was) (?:ignored|not (?:included|considered|modeled))")


def _find(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def _evidence(text: str, positive: str, negated: str) -> str:
    if _find(negated, text):
        return "contradicted"
    return "mentioned" if _find(positive, text) else "not_reported"


def closed_overlaps(text: str, profile: dict, paper_markets: list[str]) -> list[dict]:
    """Keyword matches against closed work. A match is a review prompt, not proof
    of equivalence; it only affects the route when the markets overlap."""
    hits = []
    for mech in profile["closed_mechanisms"]:
        matched = [k for k in mech["keywords"] if re.search(rf"\b{re.escape(k)}\b", text, re.IGNORECASE)]
        if matched:
            hits.append({"id": mech["id"], "disposition": mech["disposition"], "project": mech["project"],
                         "scope": mech.get("scope", ""), "matched": matched,
                         "same_market": mech.get("market") in paper_markets})
    return hits


def markets(text: str) -> list[str]:
    found = []
    if _find(_EQUITY_INDEX, text):
        found.append("equity_index")
    if _find(_CRYPTO, text):
        found.append("crypto")
    found += [name for name, pattern in _OTHER_MARKETS.items() if _find(pattern, text)]
    return found


_OPT_UNDERLYING = [          # most specific first
    ("QQQ", r"\bqqq\b"),
    ("NDX", r"\bndx\b|nasdaq[- ]100 (?:index )?options"),
    ("SPY", r"\bspy\b"),
    ("SPX", r"\bspx\b|s&p ?500 (?:index )?options"),
    ("VIX", r"vix options"),
    ("single_stock", r"(?:individual|single)[- ]stock options|stock options"),
]
_OPT_OBS = {
    "options_open_interest": r"open interest|put[- ]call (?:open interest|ratio)",
    "implied_volatility": r"implied volatility|option[- ]implied|\bskew\b|volatility (?:surface|smile)|term structure of (?:implied )?volatility",
    "options_eod_greeks": r"\bgamma\b|\bdelta\b|\bgex\b|dealer (?:positioning|inventory|hedging)|hedging (?:pressure|demand)",
    "options_signed_flow": r"option(?:s)? (?:order )?flow|signed option|customer option|option(?:s)? volume|buyer[- ]initiated option|order imbalance in options",
    "options_intraday": r"intraday option|minute[- ]level option|high[- ]frequency option|0dte|zero[- ]days? to expir|same[- ]day expir|real[- ]time option",
}
_UNVALIDATED_CONSTRUCTIONS = r"\bskew\b|volatility (?:surface|smile)|term structure|\bgex\b|gamma exposure|dealer"
_INPUT_DAILY = r"daily|end[- ]of[- ]day|\beod\b|prior[- ]day|previous[- ]day|previous trading day|lagged|closing"
_HORIZON = [("intraday", r"intraday (?:returns?|price|movements?)|within the (?:trading )?(?:day|session)|same[- ]day returns?"),
            ("next_day", r"next[- ]day|following (?:trading )?day|one[- ]day[- ]ahead|overnight"),
            ("multi_day", r"weekly|monthly|next (?:week|month)")]


def options_assessment(text: str, profile: dict) -> dict | None:
    """Options lane: separate the option underlying, the observations the paper
    needs, when those inputs are observed, and the futures prediction horizon."""
    if not _find(_OPTIONS_ANY, text):
        return None
    kinds = [k for k, pattern in _OPTIONS_LANE.items() if _find(pattern, text)]
    if not kinds:
        return None
    underlying = next((name for name, pattern in _OPT_UNDERLYING if _find(pattern, text)), "unknown")
    needs = [f for f, pattern in _OPT_OBS.items() if _find(pattern, text)]
    cadence = ("intraday" if "options_intraday" in needs else
               "daily" if _find(_INPUT_DAILY, text) else "unknown")
    horizon = next((name for name, pattern in _HORIZON if _find(pattern, text)), "unknown")
    owned = [d for d in profile["datasets"] if d.get("research_usable") and d.get("underlying")]
    owned_fields = {f for d in owned if d["underlying"] == underlying for f in d["fields"]}
    missing = [f for f in needs if f not in owned_fields]
    reasons = []
    if {"options_signed_flow", "options_intraday"} & set(needs):
        status = "requires_data"
        reasons.append("needs signed or intraday options flow; EOD open interest is not equivalent")
    elif underlying == "unknown":
        status = "unknown"
        reasons.append("option underlying not identified in title/abstract")
    elif not owned_fields:
        status = "requires_data"
        owned_names = sorted({d["underlying"] for d in owned})
        reasons.append(f"{underlying} options not owned (owned: {', '.join(owned_names) or 'none'}); "
                       "transferring the mechanism is a proposed adaptation, not an observed-data match")
    elif missing:
        status = "requires_data"
        reasons.append(f"{underlying} options owned but missing {', '.join(missing)}")
    elif cadence == "unknown":
        status = "unknown"
        reasons.append("input observation timing not stated")
    else:
        status = "potentially_compatible"
        lags = sorted({d.get("availability_lag", "") for d in owned if d["underlying"] == underlying} - {""})
        reasons.append(f"inputs look observable in owned {underlying} EOD data"
                       + (f", usable {lags[0]}" if lags else ""))
    if _find(_UNVALIDATED_CONSTRUCTIONS, text):
        reasons.append("skew/surface/term-structure/GEX constructions are not validated just because fields exist")
    return {"kinds": kinds, "underlying": underlying, "requires": needs, "input_cadence": cadence,
            "prediction_horizon": horizon, "compatibility": status, "reasons": reasons}


def proposed_instruments(paper_markets: list[str], text: str, profile: dict) -> dict[str, str]:
    """Profile instruments a paper maps to: 'named' when the paper's market is the
    instrument's market; 'transfer' for generic futures papers."""
    named = {i: "named" for i, meta in profile["instruments"].items() if meta.get("market") in paper_markets}
    if named:
        return named
    if _find(_FINANCE["futures"], text) and not [m for m in paper_markets if m not in ("equity_index", "crypto")]:
        return {i: "transfer" for i, meta in profile["instruments"].items() if meta.get("market") == "equity_index"}
    return {}


def feasibility(requirements: list[str], instruments: dict[str, str], profile: dict,
                option_underlying: str = "unknown") -> dict:
    """Per-instrument matrix: each requirement is owned / missing / unknown,
    judged only from research-usable datasets that cover that instrument."""
    matrix = {}
    for inst, basis in instruments.items():
        usable = [d for d in profile["datasets"] if d.get("research_usable")
                  and (d.get("instrument") == inst or inst in d.get("serves", []))]
        cells = {}
        for req in ["ohlcv", *requirements]:
            if req.startswith("options_") or req == "implied_volatility":
                if option_underlying == "unknown":
                    cells[req] = "unknown"
                    continue
                candidates = [d for d in usable if d.get("underlying") == option_underlying]
            else:
                candidates = usable
            cells[req] = "owned" if any(req in d.get("fields", []) for d in candidates) else "missing"
        status = ("requires_data" if "missing" in cells.values() else
                  "unknown" if "unknown" in cells.values() else "feasible")
        matrix[inst] = {"basis": basis, "status": status, "requirements": cells}
    return matrix


# Trade-frequency screen. Regex markers only: they suggest how often a
# mechanism can trade and what it reports, never what it would earn.
_ONE_PER_DAY = (r"first half[- ]hour|last half[- ]hour|opening (?:range|auction|gap)|overnight|close[- ]to[- ]close"
                r"|close[- ]to[- ]open|next[- ]day|end[- ]of[- ]day|announcements?|\bfomc\b|earnings|once (?:a|per) day")
_MULTI_PER_DAY = (r"high[- ]frequency|tick[- ]by[- ]tick|trade[- ]by[- ]trade|transaction[- ]level|\bseconds?\b"
                  r"|\b(?:1|one|5|five|10|ten|15|fifteen|30|thirty)[- ]minutes?\b|order flow|order imbalance"
                  r"|several (?:trades|times) (?:a|per) day|intraday (?:signals|trades|reversals)")
_SLOWER = r"\bdaily\b|\bweekly\b|\bmonthly\b|\bquarterly\b|rebalanc"
_BPS = r"(-?\d+(?:\.\d+)?)\s*(?:bps|basis points?|bp)\b"


def screen(text: str, profile: dict) -> dict:
    """Trade-frequency class, any basis-point figures the abstract reports, and
    flags against the profile's screen. A flag is a review prompt, not a route."""
    if _find(_ONE_PER_DAY, text):
        freq = "at_most_daily"
    elif _find(_MULTI_PER_DAY, text):
        freq = "multi_per_day"
    elif _find(_SLOWER, text):
        freq = "at_most_daily"
    else:
        freq = "unclear"
    bps = [float(x) for x in re.findall(_BPS, text, re.IGNORECASE)]
    flags = []
    bars = profile.get("screen", {}).get("min_gross_bps", {})
    if freq == "at_most_daily" and bars:
        note = profile.get("screen", {}).get("power_note", "")
        flags.append("low_frequency: at most one trade per day" + (f"; {note}" if note else ""))
    if bps and bars and max(bps) < bars["single_leg"]:
        flags.append(f"reported bps all below {bars['single_leg']} (figures may not be per-trade gross)")
    return {"trade_frequency": freq, "reported_bps": bps, "flags": flags}


def _allows_multiday(instrument: str, profile: dict) -> bool:
    """True unless every objective covering the instrument is intraday-only.
    Instruments no objective mentions are allowed."""
    horizons = [o.get("deployable_horizon", "any") for o in profile.get("objectives", [])
                if instrument in o.get("instruments", [])]
    return not horizons or any(h != "intraday_flat_by_close" for h in horizons)


def route(paper: dict, profile: dict) -> dict:
    title = paper.get("title") or ""
    abstract = paper.get("abstract_full") or paper.get("abstract") or ""
    text = f"{title}. {abstract}"
    result = {"route": "", "reasons": [], "markets": markets(text), "closed_overlaps": [],
              "requires": [], "missing_data": [], "feasibility": {}, "horizon": "unclear",
              "options_lane": options_assessment(text, profile),
              "evidence": {"out_of_sample": _evidence(abstract, _OOS, _OOS_NEG),
                           "costs": _evidence(abstract, _COST, _COST_NEG)},
              "screen": screen(text, profile)}
    if _find(_INTRADAY, text):
        result["horizon"] = "intraday"
    elif _find(_MULTIDAY, text):
        result["horizon"] = "multi_day"

    if len(abstract.split()) < 25:
        result["route"] = "insufficient_evidence"
        result["reasons"].append("no usable abstract; enrich metadata before judging")
        return result

    anchors = [name for name, pattern in _FINANCE.items() if _find(pattern, text)]
    if len(anchors) < 2:
        result["route"] = "irrelevant"
        result["reasons"].append(f"finance anchors found: {anchors or 'none'}")
        return result

    result["closed_overlaps"] = closed_overlaps(text, profile, result["markets"])
    same_market = [h for h in result["closed_overlaps"] if h["same_market"]]
    options = result["options_lane"] or {}
    result["requires"] = [name for name, pattern in _REQUIRES.items() if _find(pattern, text)]
    result["requires"] += [f for f in options.get("requires", []) if f not in result["requires"]]
    instruments = proposed_instruments(result["markets"], text, profile)
    result["feasibility"] = feasibility(result["requires"], instruments, profile,
                                        options.get("underlying", "unknown"))
    result["missing_data"] = [f"{inst}: {req}" for inst, row in result["feasibility"].items()
                              for req, cell in row["requirements"].items() if cell == "missing"]
    feasible = [i for i, row in result["feasibility"].items() if row["status"] == "feasible"]
    unknown = [i for i, row in result["feasibility"].items() if row["status"] == "unknown"]
    other = [m for m in result["markets"] if m not in ("equity_index", "crypto")]
    theoretical = _find(_THEORY, text) and not _find(_EMPIRICAL, text)

    if same_market:
        result["route"] = "overlaps_closed_work"
        result["reasons"].append("matches " + ", ".join(f"{h['id']} ({h['scope']})" for h in same_market)
                                 + "; keyword match is a review prompt - state what is genuinely different")
    elif other and not instruments:
        result["route"] = "requires_data"
        result["reasons"].append(f"evidence only on unowned markets ({', '.join(other)}); "
                                 "mechanism may transfer, needs human judgment")
    elif not instruments:
        result["route"] = "knowledge_only"
        result["reasons"].append("names no market that maps to a profile instrument")
    elif not feasible and unknown:
        result["route"] = "insufficient_evidence"
        result["reasons"].append(f"feasibility unknown for {', '.join(unknown)}: "
                                 + "; ".join(options.get("reasons", [])[:1] or ["requirements unclear"]))
    elif not feasible:
        result["route"] = "requires_data"
        result["reasons"].append("needs " + "; ".join(result["missing_data"]))
    elif _find(_PRICING, text) and not _find(_SIGNAL, abstract):
        result["route"] = "knowledge_only"
        result["reasons"].append("derivative pricing/hedging, not a trading mechanism")
    elif theoretical:
        result["route"] = "knowledge_only"
        result["reasons"].append("theoretical; no empirical test")
    elif result["horizon"] == "multi_day" and not any(_allows_multiday(i, profile) for i in feasible):
        result["route"] = "knowledge_only"
        result["reasons"].append("multi-day horizon; profile objectives for these instruments are intraday only")
    elif not _find(_SIGNAL, text):
        result["route"] = "knowledge_only"
        result["reasons"].append("no predictive or tradable claim in title/abstract")
    else:
        result["route"] = "candidate_for_review"
        basis = ", ".join(f"{i} ({result['feasibility'][i]['basis']})" for i in feasible)
        result["reasons"].append(f"observations owned for {basis}; still needs a written test plan")
        blocked = [i for i in result["feasibility"] if i not in feasible]
        if blocked:
            result["reasons"].append("not feasible on " + ", ".join(blocked) + ": "
                                     + "; ".join(m for m in result["missing_data"] if m.split(":")[0] in blocked))

    for h in result["closed_overlaps"]:
        if not h["same_market"]:
            result["reasons"].append(f"related closed work on another market: {h['id']} ({h['scope']})")
    for key, value in result["evidence"].items():
        if value == "contradicted":
            result["reasons"].append(f"abstract says {key.replace('_', ' ')} not done")
    if result["route"] == "candidate_for_review":
        result["reasons"] += result["screen"]["flags"]
    return result


ROUTE_ORDER = ["candidate_for_review", "knowledge_only", "requires_data",
               "overlaps_closed_work", "insufficient_evidence", "irrelevant"]
