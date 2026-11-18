"""
output.py — terminal printing and CSV writing.
"""
import csv
import re
import json
import os
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import config


def _kept(p: dict, min_score: int) -> bool:
    return p["kept"] if "kept" in p else p["score"] >= min_score


def print_results(papers: list[dict], min_score: int = 0, top_n: int = 20, rule: str = "") -> None:
    filtered = [p for p in papers if _kept(p, min_score)][:top_n]
    if not filtered:
        print("  No papers met the keep rule.")
        return

    print(f"\n{'='*80}")
    print(f"  TOP {len(filtered)} PAPERS  ({rule or f'score >= {min_score}'})")
    print(f"{'='*80}")
    for i, p in enumerate(filtered, 1):
        print(f"\n#{i}  [{p['score']}/10]  {p['title']}")
        if p.get("route"):
            print(f"     Route: {p['route']} - {p.get('route_reasons', '')}")
            print(f"     Trades: {p.get('trade_frequency', '')}  |  bps reported: {p.get('reported_bps') or '-'}")
        print(f"     {p['authors']}  ({p['year']})  |  {p['citations']} citations  |  source: {p['source']}")
        if p["venue"]:
            print(f"     Venue: {p['venue']}")
        print(f"     URL: {p['url'] or '(no link)'}")
        if p["oa_pdf"]:
            print(f"     PDF: {p['oa_pdf']}")
        if p.get("local_pdf"):
            print(f"     Local PDF: {p['local_pdf']}")
        print(f"     Score notes: {p['score_notes']}")
        if p["abstract"]:
            print(f"     Abstract: {p['abstract'][:200]}...")


def save_lane_csv(papers: list[dict], path: Path) -> Path:
    fields = ["score", "route", "option_compatibility", "options_lane", "option_underlying", "option_inputs",
              "option_input_cadence", "option_prediction_horizon", "option_reasons", "title", "authors", "year", "citations", "url",
              "oa_pdf", "route_reasons", "abstract", "paper_id"]
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(papers)
    return path


def create_run_dir() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ") + "_" + uuid4().hex[:8]
    path = config.RESULTS_DIR / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path: Path, data) -> None:
    """Atomically update an artifact inside a uniquely allocated run directory."""
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def save_csv(papers: list[dict], query: str, min_score: int = 0,
             run_dir: Path | None = None) -> tuple[Path, Path]:
    """
    Write two CSVs:
      _all.csv  — every paper fetched (unfiltered, for reference)
      _kept.csv — only papers meeting min_score (the actual keep-list)
    Returns (all_path, kept_path).
    """
    run_dir = run_dir if run_dir is not None else create_run_dir()
    slug   = re.sub(r"[^\w]+", "_", query.lower())[:40]
    fields = ["score", "title", "authors", "year", "citations", "inf_cites",
              "venue", "url", "ssrn_id", "doi", "source", "oa_pdf", "local_pdf",
              "score_notes", "route", "route_provisional", "route_reasons", "trade_frequency", "reported_bps",
              "screen_flags", "abstract", "paper_id", "arxiv_id",
              "fulltext_status", "draft_status"]

    all_path  = run_dir / f"{slug}_all.csv"
    kept_path = run_dir / f"{slug}_kept.csv"

    for path, rows in [(all_path, papers),
                       (kept_path, [p for p in papers if _kept(p, min_score)])]:
        with open(path, "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    return all_path, kept_path
