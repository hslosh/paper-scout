"""
sources/arxiv.py — arXiv Atom API adapter.

https://export.arxiv.org/api/query — no key needed. The shared transport
enforces config.ARXIV_RATE_LIMIT across pages, queries and retries.
"""
import xml.etree.ElementTree as ET

from .common import SearchResult, ProviderError, request

import config

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Since Sept 2026 export.arxiv.org's CDN answers cache-missing queries from
# some clients (including Python requests, while curl succeeds) with an empty
# 403/406. Header changes do not help and retries do not clear it.
_REFUSALS = {"HTTP 403", "HTTP 406"}
_REFUSAL_HINT = (" (refused by export.arxiv.org edge; arXiv papers are still"
                 " indexed via openalex/semantic_scholar)")


def search_arxiv(query: str, limit: int = config.MAX_RESULTS) -> SearchResult:
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = SearchResult()
    start = 0
    while len(result.records) < limit:
        want = min(100, limit - len(result.records))
        params = {"search_query": f"all:{query}", "start": start, "max_results": want}
        try:
            response = request("arxiv", config.ARXIV_API_BASE, config.ARXIV_RATE_LIMIT,
                               result, params=params)
            try:
                root = ET.fromstring(response.content)
            finally:
                response.close()
            if root.tag != "{http://www.w3.org/2005/Atom}feed":
                raise ProviderError("invalid Atom response")
            entries = root.findall("atom:entry", _NS)
            batch = [_parse_entry(entry) for entry in entries]
            if any(not r["title"] or not r["paperId"] or '/api/errors' in _text(e, "id")
                   for r, e in zip(batch, entries)):
                raise ProviderError("invalid Atom paper or API error entry")
            result.records.extend(batch[:want])
            result.pages += 1
            result.limit_reached = len(result.records) >= limit
            if len(entries) < want or result.limit_reached:
                break
            start += len(entries)
        except (ProviderError, ET.ParseError, ValueError, TypeError, AttributeError) as exc:
            result.error = str(exc) if isinstance(exc, ProviderError) else "invalid Atom response"
            if result.error in _REFUSALS:
                result.error += _REFUSAL_HINT
            break
    return result


def _text(entry, tag: str) -> str:
    return (entry.findtext(f"atom:{tag}", default="", namespaces=_NS) or "").strip()


def _parse_entry(entry) -> dict:
    title    = _text(entry, "title").replace("\n", " ").strip()
    abstract = _text(entry, "summary").replace("\n", " ").strip()
    published = _text(entry, "published")
    year = int(published[:4]) if published[:4].isdigit() else None

    authors = [{"name": (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()}
               for a in entry.findall("atom:author", _NS)]

    id_url   = _text(entry, "id")
    arxiv_id = id_url.split("/abs/", 1)[-1] if "/abs/" in id_url else ""

    pdf_url = ""
    for link in entry.findall("atom:link", _NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "")
            break

    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "citationCount": 0,            # arXiv's API doesn't expose citation counts
        "influentialCitationCount": 0,
        "publicationDate": published[:10] if published else "",
        "externalIds": {"ArXiv": arxiv_id},
        "openAccessPdf": {"url": pdf_url},   # arXiv preprints are open by definition
        "publicationVenue": {"name": "arXiv"},
        "authors": authors,
        "paperId": f"arxiv:{arxiv_id}" if arxiv_id else id_url,
        "source": "arxiv",
    }
