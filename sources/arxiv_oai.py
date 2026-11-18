"""
sources/arxiv_oai.py — local arXiv index harvested via OAI-PMH.

export.arxiv.org/api refuses Python clients (HTTP 406, Sept 2026), but the
OAI-PMH endpoint is arXiv's sanctioned bulk-metadata channel and works. We
harvest configured sets (default: q-fin) incrementally into SQLite and search
them locally with FTS5/bm25. `search_arxiv_index` returns records in the
shared raw schema, so identity merge and scoring treat them like any source.
"""
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import SearchResult, ProviderError, request

import config

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_AX = "{http://arxiv.org/OAI/arXiv/}"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT, authors TEXT,
    categories TEXT, doi TEXT, created TEXT, updated TEXT, datestamp TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
    USING fts5(id UNINDEXED, title, abstract, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path or config.ARXIV_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _first_month(arxiv_id: str) -> str:
    """YYYY-MM of first submission, encoded in the ID. OAI `created` is not
    reliable for this (it can be a later version's date)."""
    m = re.match(r"(?:[a-z\-]+(?:\.[A-Z]{2})?/)?(\d{2})(\d{2})", arxiv_id)
    if not m:
        return ""
    yy = int(m.group(1))
    return f"{2000 + yy if yy < 90 else 1900 + yy}-{m.group(2)}"


def parse_records(root: ET.Element) -> tuple[list[dict], list[str], str, str]:
    """Return (papers, deleted_ids, max_datestamp, resumption_token)."""
    error = root.find(f"{_OAI}error")
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], [], "", ""
        raise ProviderError(f"OAI error {error.get('code')}")
    list_records = root.find(f"{_OAI}ListRecords")
    if list_records is None:
        raise ProviderError("invalid OAI response")
    papers, deleted, max_stamp = [], [], ""
    for record in list_records.findall(f"{_OAI}record"):
        header = record.find(f"{_OAI}header")
        stamp = header.findtext(f"{_OAI}datestamp", "") if header is not None else ""
        max_stamp = max(max_stamp, stamp)
        oai_id = header.findtext(f"{_OAI}identifier", "") if header is not None else ""
        if header is not None and header.get("status") == "deleted":
            deleted.append(oai_id.removeprefix("oai:arXiv.org:"))
            continue
        meta = record.find(f"{_OAI}metadata/{_AX}arXiv")
        if meta is None:
            raise ProviderError("invalid OAI record")
        arxiv_id = _clean(meta.findtext(f"{_AX}id"))
        title = _clean(meta.findtext(f"{_AX}title"))
        if not arxiv_id or not title:
            raise ProviderError("invalid OAI record")
        authors = [_clean(f"{a.findtext(f'{_AX}forenames', '')} {a.findtext(f'{_AX}keyname', '')}")
                   for a in meta.findall(f"{_AX}authors/{_AX}author")]
        papers.append({
            "id": arxiv_id, "title": title,
            "abstract": _clean(meta.findtext(f"{_AX}abstract")),
            "authors": json.dumps([a for a in authors if a], ensure_ascii=False),
            "categories": _clean(meta.findtext(f"{_AX}categories")),
            "doi": _clean(meta.findtext(f"{_AX}doi")).split(" ")[0],
            "created": _clean(meta.findtext(f"{_AX}created")),
            "updated": _clean(meta.findtext(f"{_AX}updated")),
            "datestamp": stamp,
        })
    token = list_records.findtext(f"{_OAI}resumptionToken", "") or ""
    return papers, deleted, max_stamp, token.strip()


def _store(conn: sqlite3.Connection, papers: list[dict], deleted: list[str]) -> None:
    ids = [p["id"] for p in papers] + deleted
    conn.executemany("DELETE FROM papers_fts WHERE id = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM papers WHERE id = ?", [(i,) for i in deleted])
    conn.executemany(
        "INSERT OR REPLACE INTO papers VALUES "
        "(:id, :title, :abstract, :authors, :categories, :doi, :created, :updated, :datestamp)", papers)
    conn.executemany("INSERT INTO papers_fts (id, title, abstract) VALUES (:id, :title, :abstract)", papers)


def sync(sets: list[str] | None = None, db_path: Path | None = None, log=print) -> dict:
    """Incrementally harvest OAI sets; commits after every page so an
    interrupted sync resumes from the last stored datestamp."""
    sets = sets or config.ARXIV_OAI_SETS
    conn = connect(db_path)
    summary = {}
    try:
        for oai_set in sets:
            result = SearchResult()
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (f"from:{oai_set}",)).fetchone()
            params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set}
            if row:
                params["from"] = row[0]
            stored = 0
            try:
                while True:
                    response = request("arxiv_oai", config.ARXIV_OAI_BASE, config.ARXIV_RATE_LIMIT,
                                       result, params=params, timeout=120)
                    try:
                        root = ET.fromstring(response.content)
                    finally:
                        response.close()
                    papers, deleted, stamp, token = parse_records(root)
                    with conn:
                        _store(conn, papers, deleted)
                        if stamp and not token:
                            conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"from:{oai_set}", stamp))
                    stored += len(papers)
                    result.pages += 1
                    log(f"  [arxiv sync] {oai_set}: page {result.pages}, {stored} records")
                    if not token:
                        break
                    params = {"verb": "ListRecords", "resumptionToken": token}
            except (ProviderError, ET.ParseError) as exc:
                result.error = str(exc) if isinstance(exc, ProviderError) else "invalid OAI XML"
            summary[oai_set] = {"stored": stored, "pages": result.pages,
                                "retries": result.retries, "error": result.error}
    finally:
        conn.close()
    return summary


def _fts_terms(query: str) -> list[str]:
    return [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1]


def search_arxiv_index(query: str, limit: int = config.MAX_RESULTS,
                       db_path: Path | None = None) -> SearchResult:
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = SearchResult()
    db_path = Path(db_path or config.ARXIV_DB)
    if not db_path.exists():
        result.error = "arXiv index not built; run with --sync-arxiv"
        return result
    terms = _fts_terms(query)
    if not terms:
        return result
    conn = connect(db_path)
    try:
        if conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0:
            result.error = "arXiv index is empty; run with --sync-arxiv"
            return result
        rows, seen = [], set()
        # All terms first (closest to the API's behaviour), then any term to fill.
        for joiner in (" AND ", " OR "):
            if len(rows) >= limit:
                break
            match = joiner.join(f'"{t}"' for t in terms)
            for row in conn.execute(
                    "SELECT p.id, p.title, p.abstract, p.authors, p.doi, p.created "
                    "FROM papers_fts f JOIN papers p ON p.id = f.id "
                    "WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts, 0, 3.0, 1.0) LIMIT ?",
                    (match, limit)):
                if row[0] not in seen and len(rows) < limit:
                    seen.add(row[0])
                    rows.append(row)
    except sqlite3.Error as exc:
        result.error = f"arXiv index error: {type(exc).__name__}"
        return result
    finally:
        conn.close()
    for arxiv_id, title, abstract, authors, doi, created in rows:
        month = _first_month(arxiv_id)
        ids = {"ArXiv": arxiv_id}
        if doi:
            ids["DOI"] = doi
        result.records.append({
            "title": title, "abstract": abstract or "",
            "year": int(month[:4]) if month else (int(created[:4]) if (created or "")[:4].isdigit() else None),
            "citationCount": 0, "influentialCitationCount": 0,
            "publicationDate": month, "externalIds": ids,
            "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv_id}"},
            "publicationVenue": {"name": "arXiv"},
            "authors": [{"name": a} for a in json.loads(authors or "[]")],
            "paperId": f"arxiv:{arxiv_id}", "source": "arxiv",
        })
    result.pages = 1
    result.limit_reached = len(result.records) >= limit
    return result
