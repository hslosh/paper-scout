"""
paper_scout.py — Paper Scout
===========================
Searches Semantic Scholar, arXiv, and OpenAlex for quantitative trading
papers and scores them for backtest-worthiness. See README.md for full
setup/usage; this docstring covers the essentials.

Usage:
    python3 paper_scout.py                              # default queries, all sources
    python3 paper_scout.py "momentum futures"            # custom query
    python3 paper_scout.py --min-score 6 --sources arxiv,openalex
    python3 paper_scout.py --profile microstructure --save-vault --draft-hypothesis

Output:
    - Ranked results printed to terminal
    - Two CSVs per unique results/runs/<run-id>/ directory: _all and _kept
      (>= min_score), tagged with each paper's source
    - Optional: full-text PDFs (via Unpaywall) for kept papers -> results/pdfs/
    - Optional: Obsidian source notes for top papers, with an LLM-drafted
      trading hypothesis in "Key Ideas" (via a local Ollama model)

Scoring: see scoring.py (profile-aware; `general` and `microstructure`).
"""
import argparse
import sys
from datetime import datetime, timezone
import hashlib

import config
from scoring import score_abstract
from sources import SEARCH_FUNCS, arxiv_oai
from output import print_results, save_csv, save_lane_csv, create_run_dir, save_json
from identity import merge_records, normalize_doi
from vault import save_obsidian_note
from fulltext import fetch_fulltext_pdf, extract_pdf_text
from llm_draft import draft_hypothesis
import evidence
from identity import file_id
import research_profile
import routing

# Windows consoles default stdout to the system codepage (e.g. cp1252), which
# can't encode non-ASCII characters that show up in author names/abstracts
# (accents, special quotes, etc.) and crashes mid-run. Force UTF-8 with a
# replace fallback so a single unprintable character never kills the whole
# scout run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_paper(raw: dict, profile: str = "general") -> dict | None:
    """Normalize a raw record (from any source adapter) into our schema."""
    title    = (raw.get("title") or "").strip()
    if not title:
        return None

    abstract = raw.get("abstract") or ""
    year     = raw.get("year") or 0
    cites    = raw.get("citationCount") or 0
    inf_cites= raw.get("influentialCitationCount") or 0
    pub_date = raw.get("publicationDate") or ""
    source   = raw.get("source", "semantic_scholar")

    ext_ids  = raw.get("externalIds") or {}
    ssrn_id  = ext_ids.get("SSRN")
    doi      = normalize_doi(ext_ids.get("DOI") or "")
    arxiv_id = ext_ids.get("ArXiv")

    if ssrn_id:
        url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={ssrn_id}"
    elif doi:
        url = f"https://doi.org/{doi}"
    elif arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    else:
        url = ""

    oa_pdf  = (raw.get("openAccessPdf") or {}).get("url", "")
    venue   = (raw.get("publicationVenue") or {}).get("name", "")
    authors = ", ".join(a.get("name", "") for a in (raw.get("authors") or [])[:3])
    if len(raw.get("authors") or []) > 3:
        authors += " et al."

    score, notes = score_abstract(abstract, year, cites, oa_pdf=oa_pdf,
                                   ssrn_id=ssrn_id or "", profile=profile)

    return {
        "score"      : score,
        "title"      : title,
        "authors"    : authors,
        "year"       : year,
        "pub_date"   : pub_date,
        "citations"  : cites,
        "inf_cites"  : inf_cites,
        "venue"      : venue,
        "url"        : url,
        "ssrn_id"    : ssrn_id or "",
        "doi"        : doi or "",
        "arxiv_id"   : arxiv_id or "",
        "source"     : source,
        "oa_pdf"     : oa_pdf,
        "local_pdf"  : "",   # filled in later if --draft-hypothesis / full-text retrieval runs
        "abstract"   : abstract[:600] + ("..." if len(abstract) > 600 else ""),
        "abstract_full": abstract,   # kept for Unpaywall/LLM use; not written to CSV
        "score_notes": " | ".join(notes),
        "paper_id"   : raw.get("paperId", ""),
        "sources"    : raw.get("sources", [source]),
        "queries"    : raw.get("queries", []),
        "provenance" : raw.get("provenance", []),
        "oa_pdfs"    : raw.get("oa_pdfs", []),
        "fulltext_status": "not_requested",
        "draft_status": "not_requested",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Scout for backtest-worthy trading papers")
    parser.add_argument("query", nargs="?", default=None,
                        help="Search query (default: runs the built-in query set for --query-profile)")
    parser.add_argument("--min-score", type=int, default=5,
                        help="Minimum score to display/keep (0-9+, default 5)")
    parser.add_argument("--top", type=int, default=15, help="Max results to show")
    parser.add_argument("--save-vault", action="store_true",
                        help="Save top-3 kept papers as markdown notes (VAULT_PATH)")
    parser.add_argument("--limit", type=int, default=config.MAX_RESULTS,
                        help="API results to fetch per query per source")
    parser.add_argument("--sources", type=str, default=",".join(config.DEFAULT_SOURCES),
                        help=f"Comma-separated sources to query: {','.join(config.ALL_SOURCES)}")
    parser.add_argument("--profile", type=str, default="general", choices=["general", "microstructure"],
                        help="Scoring profile (default: general); also the default query set")
    parser.add_argument("--query-profile", type=str, default=None, choices=sorted(config.QUERY_PROFILES),
                        help="Built-in query set, independent of scoring (default: same as --profile). "
                             "options_eod / options_flow are search coverage only; scoring is unchanged")
    parser.add_argument("--fetch-fulltext", action="store_true",
                        help="Download provider-declared OA PDFs, with Unpaywall fallback, for all kept papers")
    parser.add_argument("--draft-hypothesis", action="store_true",
                        help="Draft top kept papers via local Ollama; always save drafts in the run directory")
    parser.add_argument("--draft-limit", type=int, default=3,
                        help="Maximum papers to draft or card (default 3)")
    parser.add_argument("--evidence-cards", action="store_true",
                        help="Build verified evidence cards for the top kept papers (local Ollama); "
                             "saved under the run's cards/ directory")
    parser.add_argument("--cards-from", choices=["kept", "options"], default="kept",
                        help="Which list --evidence-cards draws from: the keep-list (default) or the options "
                             "research lane (potentially_compatible first; routes are not changed)")
    parser.add_argument("--select", choices=["route", "score"], default="route",
                        help="Keep-list rule: 'route' keeps candidate_for_review papers from the research "
                             "profile (default); 'score' keeps papers with score >= --min-score")
    parser.add_argument("--sync-arxiv", action="store_true",
                        help="Incrementally harvest arXiv OAI-PMH sets into the local index before searching")
    args = parser.parse_args(argv)

    sources = list(dict.fromkeys(s.strip() for s in args.sources.split(",") if s.strip()))
    if not sources:
        parser.error("at least one source is required")
    if args.limit <= 0 or args.top <= 0 or args.draft_limit <= 0:
        parser.error("--limit, --top and --draft-limit must be positive")
    if args.min_score < 0:
        parser.error("--min-score must be nonnegative")
    if args.query is not None and not args.query.strip():
        parser.error("query must not be blank")
    if "semantic_scholar" in sources and args.limit > 1000:
        parser.error("Semantic Scholar supports --limit up to 1000")
    unknown = [s for s in sources if s not in SEARCH_FUNCS]
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)} (choices: {', '.join(config.ALL_SOURCES)})")

    query_profile = args.query_profile or args.profile
    queries = [args.query] if args.query else config.QUERY_PROFILES[query_profile]

    run_dir = create_run_dir()
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "arguments": vars(args), "searches": [],
                "ollama_model": config.OLLAMA_MODEL, "artifact_errors": []}
    code_files = sorted(config.ROOT_DIR.glob("*.py")) + sorted((config.ROOT_DIR / "sources").glob("*.py"))
    manifest["code_sha256"] = {str(p.relative_to(config.ROOT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in code_files}
    save_json(run_dir / "manifest.json", manifest)
    if args.sync_arxiv:
        print("\n[ARXIV SYNC] harvesting OAI-PMH sets ...")
        manifest["arxiv_sync"] = arxiv_oai.sync()
        save_json(run_dir / "manifest.json", manifest)
        if any(s["error"] for s in manifest["arxiv_sync"].values()):
            manifest["artifact_errors"].append("arXiv sync incomplete")
    raw_records = []

    for q in queries:
        print(f"\n[SEARCHING] '{q}' ...")
        for src in sources:
            print(f"  [{src}]")
            result = SEARCH_FUNCS[src](q, limit=args.limit)
            manifest["searches"].append({"source": src, "query": q,
                                         "retrieved_at": datetime.now(timezone.utc).isoformat(), **result.summary()})
            print(f"    {result.status}: {len(result.records)} records" +
                  (f" ({result.error})" if result.error else ""))
            raw_records.extend(dict(r, source=src, query=q) for r in result.records)
            save_json(run_dir / "raw_records.json", raw_records)
            save_json(run_dir / "manifest.json", manifest)

    deduped = []
    for raw in merge_records(raw_records):
        try:
            parsed = parse_paper(raw, profile=args.profile)
            if parsed is None:
                raise ValueError("missing title")
            deduped.append(parsed)
        except (ValueError, TypeError, AttributeError):
            manifest["artifact_errors"].append("invalid paper metadata during normalization")

    research = None
    try:
        research = research_profile.load()
        manifest["research_profile"] = research_profile.status(research)
    except research_profile.ProfileError as exc:
        manifest["artifact_errors"].append(f"research profile: {exc}")
    for p in deduped:
        if research is not None:
            r = routing.route(p, research)
            p.update(route=r["route"], route_reasons=" | ".join(r["reasons"]), routing=r,
                     route_provisional=manifest["research_profile"]["provisional"],
                     trade_frequency=r["screen"]["trade_frequency"],
                     reported_bps=" ".join(f"{x:g}" for x in r["screen"]["reported_bps"]),
                     screen_flags=" | ".join(r["screen"]["flags"]))
        else:
            p.update(route="", route_reasons="", routing={}, route_provisional=True)
    select = args.select if research is not None else "score"
    manifest["selection"] = select
    if select == "route":
        rank = {name: i for i, name in enumerate(routing.ROUTE_ORDER)}
        # Within a route, mechanisms that can trade several times a day sort first.
        freq_rank = {"multi_per_day": 0, "unclear": 1, "at_most_daily": 2}
        deduped.sort(key=lambda x: (rank.get(x["route"], 99), freq_rank.get(x.get("trade_frequency"), 1),
                                    -x["score"], -x["citations"]))
    else:
        deduped.sort(key=lambda x: (x["score"], x["citations"]), reverse=True)
    for p in deduped:
        p["kept"] = (p["route"] == "candidate_for_review" if select == "route"
                     else p["score"] >= args.min_score)

    print(f"\n[RESULTS] {len(deduped)} unique papers across {len(queries)} queries, "
          f"{len(sources)} source(s)")

    # Full-text retrieval for kept papers (needed before hypothesis drafting can
    # use full text instead of just the abstract).
    kept = [p for p in deduped if p["kept"]]
    draft_papers = kept[:args.draft_limit] if args.draft_hypothesis else []
    # The options lane keeps each paper's own route; carding it does not promote it.
    options = [p for p in deduped if (p.get("routing") or {}).get("options_lane")
               and p["route"] not in ("irrelevant", "insufficient_evidence")]
    compat_rank = {"potentially_compatible": 0, "unknown": 1, "requires_data": 2}
    card_pool = kept if args.cards_from == "kept" else sorted(
        options, key=lambda p: compat_rank.get(p["routing"]["options_lane"]["compatibility"], 3))
    card_papers = card_pool[:args.draft_limit] if args.evidence_cards else []
    fetch_papers = kept if args.fetch_fulltext else list({id(p): p for p in draft_papers + card_papers}.values())
    save_json(run_dir / "papers.json", deduped)
    if fetch_papers:
        print(f"\n[FULLTEXT] resolving OA PDFs for {len(fetch_papers)} paper(s)...")
        for p in fetch_papers:
            local_pdf = fetch_fulltext_pdf(p)
            if local_pdf:
                p["local_pdf"] = local_pdf
                print(f"  [FULLTEXT] {p['title'][:60]}... -> {local_pdf}")
            save_json(run_dir / "papers.json", deduped)

    drafts = []
    for p in draft_papers:
        full_text = extract_pdf_text(p["local_pdf"]) if p.get("local_pdf") else ""
        p["draft_input"] = "pdf_excerpt" if full_text else "abstract"
        hypothesis = draft_hypothesis(p, full_text=full_text)
        p["hypothesis"] = hypothesis
        p["draft_status"] = ("failed" if hypothesis.startswith("[OLLAMA ERROR]") else
                             "skipped" if hypothesis.startswith("[OLLAMA SKIP]") else "complete")
        drafts.append({"paper_id": p["paper_id"], "doi": p["doi"], "title": p["title"],
                       "input": p["draft_input"], "status": p["draft_status"], "draft": hypothesis})
        save_json(run_dir / "drafts.json", drafts)
        save_json(run_dir / "papers.json", deduped)
    if args.draft_hypothesis and not drafts:
        save_json(run_dir / "drafts.json", [])

    if card_papers:
        (run_dir / "cards").mkdir()
        print(f"\n[CARDS] extracting evidence for {len(card_papers)} paper(s) with {config.OLLAMA_MODEL} ...")
    for p in card_papers:
        text, pages_used = evidence.select_text(p, evidence.pdf_pages(p.get("local_pdf", "")))
        card = evidence.build_card(p, evidence.extract(p["title"], text), text, pages_used)
        p["card_status"] = card["status"]
        stem = run_dir / "cards" / file_id(p)
        save_json(stem.with_suffix(".json"), card)
        stem.with_suffix(".md").write_text(evidence.card_markdown(card), encoding="utf-8")
        ver = card.get("verification") or {}
        print(f"  [CARD] {card['status']}: {p['title'][:60]} "
              f"(quotes verified {ver.get('verified', 0)}/{ver.get('quotes', 0)})")
        if card["status"] != "failed":
            p.setdefault("hypothesis", evidence.card_markdown(card))
        save_json(run_dir / "papers.json", deduped)

    rule = "route = candidate_for_review" if select == "route" else f"score >= {args.min_score}"
    print_results(deduped, top_n=args.top, rule=rule)

    query_label = args.query or f"{query_profile}_multi_query"
    all_path, kept_path = save_csv(deduped, query_label, run_dir=run_dir)
    for p in options:
        lane = p["routing"]["options_lane"]
        p["options_lane"] = "+".join(lane["kinds"])
        p.update(option_underlying=lane["underlying"], option_inputs="+".join(lane["requires"]),
                 option_input_cadence=lane["input_cadence"], option_prediction_horizon=lane["prediction_horizon"],
                 option_compatibility=lane["compatibility"], option_reasons=" | ".join(lane["reasons"]))
    if options:
        options_path = save_lane_csv(options, run_dir / "options_research.csv")
        print(f"[SAVED] {len(options)} options-research papers -> {options_path.name}")
    manifest["options_research"] = len(options)
    kept_count = len(kept)
    print(f"\n[SAVED] {len(deduped)} total  -> {all_path.name}")
    print(f"[SAVED] {kept_count} kept ({rule})  -> {kept_path.name}")

    if args.save_vault:
        for p in kept[:3]:
            try:
                note_path = save_obsidian_note(p, query_label, hypothesis=p.get("hypothesis", ""))
                print(f"[VAULT] preserved/created -> {note_path}")
            except OSError as exc:
                manifest["artifact_errors"].append(f"vault export: {type(exc).__name__}")

    searches = manifest["searches"]
    all_failed = all(s["status"] == "failed" for s in searches)
    incomplete = any(s["status"] in ("partial", "failed") for s in searches)
    optional_failed = any(p["fulltext_status"] == "failed" or p["draft_status"] in ("failed", "skipped")
                          or p.get("card_status") == "failed" for p in deduped)
    manifest["status"] = ("failed" if all_failed else "partial" if
                          incomplete or optional_failed or manifest["artifact_errors"] else "complete")
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["unique_papers"] = len(deduped)
    save_json(run_dir / "papers.json", deduped)
    save_json(run_dir / "manifest.json", manifest)
    if manifest.get("research_profile", {}).get("state") != "approved":
        print(f"[PROFILE] research profile is {manifest.get('research_profile', {}).get('state', 'unavailable')}; "
              "routes are unreviewed triage")
    print(f"[RUN] {manifest['status']} -> {run_dir}")
    return 1 if all_failed else 2 if manifest["status"] == "partial" else 0


if __name__ == "__main__":
    sys.exit(main())
