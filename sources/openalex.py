"""
sources/openalex.py — OpenAlex Works API adapter.

https://api.openalex.org/works — optional OPENALEX_KEY authentication.
Cursor paging uses a constant page size and reports incomplete coverage.
"""
from .common import SearchResult, ProviderError, get_json, require_records, validate_paper

import config


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [position, ...]} to dodge copyright
    concerns over full-text redistribution. Invert back to plain text."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, limit: int = config.MAX_RESULTS) -> SearchResult:
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = SearchResult()
    per_page = min(100, limit)
    cursor = "*"
    seen_cursors = set()
    while len(result.records) < limit:
        params = {"search": query, "per-page": per_page, "cursor": cursor}
        headers = {}
        if config.OPENALEX_KEY:
            headers["Authorization"] = f"Bearer {config.OPENALEX_KEY}"
        try:
            data = get_json("openalex", config.OPENALEX_API_BASE,
                            config.OPENALEX_RATE_LIMIT, result, params=params, headers=headers)
            works = require_records(data, "results")
            batch = [validate_paper(_parse_work(w)) for w in works]
            if any(not r["title"] or not r["paperId"] for r in batch):
                raise ProviderError("invalid paper metadata")
            result.records.extend(batch[:limit - len(result.records)])
            result.pages += 1
            result.limit_reached = len(result.records) >= limit
            if not works or result.limit_reached:
                break
            cursor = (data.get("meta") or {}).get("next_cursor")
            if not cursor:
                break
            if cursor in seen_cursors:
                raise ProviderError("repeated pagination cursor")
            seen_cursors.add(cursor)
        except (ProviderError, ValueError, TypeError, KeyError, AttributeError) as exc:
            result.error = str(exc) if isinstance(exc, ProviderError) else "invalid paper metadata"
            break
    return result


def _parse_work(w: dict) -> dict:
    title    = w.get("title") or w.get("display_name") or ""
    abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
    year     = w.get("publication_year")
    cites    = w.get("cited_by_count") or 0

    doi_raw = w.get("doi") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

    authors = [{"name": (a.get("author") or {}).get("display_name", "")}
               for a in (w.get("authorships") or [])]

    oa_url = ((w.get("best_oa_location") or {}).get("pdf_url")) or ""
    venue  = (((w.get("primary_location") or {}).get("source") or {}).get("display_name")) or ""

    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "citationCount": cites,
        "influentialCitationCount": 0,   # OpenAlex has no equivalent metric
        "publicationDate": w.get("publication_date") or "",
        "externalIds": {"DOI": doi},
        "openAccessPdf": {"url": oa_url},
        "publicationVenue": {"name": venue},
        "authors": authors,
        "paperId": w.get("id", ""),
        "source": "openalex",
    }
