"""
evaluate.py — replay archived results and score routing against
human labels. Never touches market data or backtest returns.

    python evaluate.py replay                 # route every archived paper; compare with old keep-lists
    python evaluate.py make-labels --n 60     # blind, stratified labeling sheet (no predictions shown)
    python evaluate.py score results/labels.csv

Labels come from a human, not the tool: routes are proposed by rules
written in this repo, so self-labeling would be circular. make-labels hides
predicted routes to avoid anchoring; score reveals them.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import config
import research_profile
import routing

LABEL_COLUMNS = ["item_id", "title", "abstract", "year", "url", "human_route", "human_worth_reading", "notes"]


def archived_papers() -> list[dict]:
    """Every paper from legacy CSVs and run papers.json files, deduplicated by title."""
    papers, seen = [], set()

    def add(p, origin):
        key = " ".join((p.get("title") or "").casefold().split())
        if key and key not in seen:
            seen.add(key)
            papers.append(dict(p, origin=origin))

    for path in sorted(config.RESULTS_DIR.glob("*_all.csv")):
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["score"] = int(row.get("score") or 0)
                add(row, path.name)
    for path in sorted((config.RESULTS_DIR / "runs").glob("*/papers.json")):
        try:
            for p in json.loads(path.read_text(encoding="utf-8")):
                add(p, path.parent.name)
        except (OSError, ValueError):
            continue
    return papers


def replay(min_score: int = 5) -> dict:
    profile = research_profile.load()
    papers = archived_papers()
    table = defaultdict(Counter)
    for p in papers:
        r = routing.route(p, profile)
        p["_route"] = r["route"]
        table["old_kept" if p["score"] >= min_score else "old_dropped"][r["route"]] += 1
    summary = {"papers": len(papers),
               "routes": dict(Counter(p["_route"] for p in papers)),
               "old_kept_rerouted": dict(table["old_kept"]),
               "old_dropped_rerouted": dict(table["old_dropped"])}
    old_kept_irrelevant = [p["title"] for p in papers if p["score"] >= min_score and p["_route"] == "irrelevant"]
    summary["old_kept_now_irrelevant_examples"] = old_kept_irrelevant[:10]
    return summary


def make_labels(out: Path, n: int, seed: int = 42) -> Path:
    profile = research_profile.load()
    papers = [p for p in archived_papers() if (p.get("abstract_full") or p.get("abstract"))]
    by_route = defaultdict(list)
    for p in papers:
        by_route[routing.route(p, profile)["route"]].append(p)
    rng = random.Random(seed)
    per = max(1, n // len(routing.ROUTE_ORDER))
    sample = []
    for name in routing.ROUTE_ORDER:                    # stratified so rare routes get labels
        group = by_route.get(name, [])
        sample += rng.sample(group, min(per, len(group)))
    rng.shuffle(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for i, p in enumerate(sample, 1):
            w.writerow({"item_id": i, "title": p["title"], "year": p.get("year", ""), "url": p.get("url", ""),
                        "abstract": (p.get("abstract_full") or p.get("abstract") or "")[:1500]})
    return out


def score(labels: Path) -> dict:
    profile = research_profile.load()
    with labels.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("human_route", "").strip()]
    if not rows:
        raise SystemExit("no labelled rows (fill human_route)")
    unknown = {r["human_route"] for r in rows} - set(routing.ROUTE_ORDER)
    if unknown:
        raise SystemExit(f"unknown human_route values: {sorted(unknown)}; use {routing.ROUTE_ORDER}")
    confusion = defaultdict(Counter)
    for r in rows:
        pred = routing.route({"title": r["title"], "abstract_full": r["abstract"]}, profile)["route"]
        confusion[r["human_route"]][pred] += 1
    agree = sum(confusion[k][k] for k in confusion)
    cand_pred = sum(c["candidate_for_review"] for c in confusion.values())
    cand_true = sum(confusion["candidate_for_review"].values())
    tp = confusion["candidate_for_review"]["candidate_for_review"]
    return {"labelled": len(rows), "agreement": round(agree / len(rows), 3),
            "candidate_precision": round(tp / cand_pred, 3) if cand_pred else None,
            "candidate_recall": round(tp / cand_true, 3) if cand_true else None,
            "confusion_human_to_predicted": {k: dict(v) for k, v in confusion.items()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replay")
    mk = sub.add_parser("make-labels")
    mk.add_argument("--n", type=int, default=60)
    mk.add_argument("--out", type=Path, default=config.RESULTS_DIR / "labels.csv")
    sc = sub.add_parser("score")
    sc.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "replay":
        result = replay()
    elif args.cmd == "make-labels":
        result = {"written": str(make_labels(args.out, args.n))}
    else:
        result = score(args.labels)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
