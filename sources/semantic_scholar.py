"""
sources/semantic_scholar.py — Semantic Scholar Graph API adapter.

Indexes SSRN among many other venues. Free key raises the rate limit from
100 req/5min (unauthenticated) to a sustained 1 req/sec.
Get one at https://www.semanticscholar.org/product/api
"""
from .common import SearchResult, ProviderError, get_json, require_records, validate_paper

import config


def search_semantic_scholar(query: str, limit: int = config.MAX_RESULTS) -> SearchResult:
    if not 1 <= limit <= 1000:
        raise ValueError("Semantic Scholar limit must be between 1 and 1000")
    result = SearchResult()
    headers = {"User-Agent": "paper-scout/1.1 (academic use)"}
    if config.SEMANTIC_SCHOLAR_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_KEY
    offset = 0
    while len(result.records) < limit:
        want = min(100, limit - len(result.records))
        params = {"query": query, "fields": config.SS_FIELDS, "limit": want, "offset": offset}
        try:
            data = get_json("semantic_scholar", f"{config.SS_API_BASE}/paper/search",
                            config.SS_RATE_LIMIT, result, params=params, headers=headers)
            batch = [validate_paper(r) for r in require_records(data, "data")]
            if any(not r.get("title") or not r.get("paperId") for r in batch):
                raise ProviderError("invalid paper metadata")
            result.records.extend(dict(r, source="semantic_scholar") for r in batch[:want])
            result.pages += 1
            result.limit_reached = len(result.records) >= limit
            if not batch or result.limit_reached:
                break
            next_offset = data.get("next")
            if next_offset is None:
                break
            if not isinstance(next_offset, int) or next_offset <= offset:
                raise ProviderError("invalid pagination offset")
            offset = next_offset
        except (ProviderError, ValueError, TypeError, KeyError, AttributeError) as exc:
            result.error = str(exc) if isinstance(exc, ProviderError) else "invalid paper metadata"
            break
    return result
