"""Sequential provider transport and explicit search outcomes.

Rate limits span queries/pages/retries. Errors deliberately omit request URLs
and response bodies, which may contain credentials or contact information.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time

import requests


@dataclass
class SearchResult:
    records: list[dict] = field(default_factory=list)
    pages: int = 0
    retries: int = 0
    error: str = ""
    limit_reached: bool = False

    @property
    def status(self) -> str:
        if self.error:
            return "partial" if self.records else "failed"
        return "complete" if self.records else "complete_empty"

    def summary(self) -> dict:
        return {"status": self.status, "records": len(self.records),
                "pages": self.pages, "retries": self.retries,
                "error": self.error, "limit_reached": self.limit_reached}


class ProviderError(Exception):
    pass


_sessions: dict[str, requests.Session] = {}
_last_request: dict[str, float] = {}


def retry_delay(value: str | None, fallback: float) -> float:
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            timestamp = parsedate_to_datetime(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            delay = (timestamp - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            delay = fallback
    if not 0 <= delay < float("inf"):
        return fallback
    return delay


def request(provider: str, url: str, interval: float, result: SearchResult,
            timeout: float = 20, **kwargs) -> requests.Response:
    if provider not in _sessions:
        _sessions[provider] = requests.Session()
    session = _sessions[provider]
    headers = {"User-Agent": "paper-scout/1.1 (academic literature search)",
               **kwargs.pop("headers", {})}
    for attempt in range(3):
        wait = interval - (time.monotonic() - _last_request.get(provider, -float("inf")))
        if wait > 0:
            time.sleep(wait)
        _last_request[provider] = time.monotonic()
        try:
            response = session.get(url, timeout=timeout, headers=headers, **kwargs)
        except requests.RequestException as exc:
            if attempt == 2:
                raise ProviderError(type(exc).__name__) from None
            delay = 2 ** attempt
        else:
            if response.status_code not in (429, 500, 502, 503, 504):
                if not 200 <= response.status_code < 300:
                    status = response.status_code
                    response.close()
                    raise ProviderError(f"HTTP {status}")
                return response
            status = response.status_code
            delay = retry_delay(response.headers.get("Retry-After"), 2 ** attempt)
            response.close()
            if attempt == 2 or delay > 60:
                raise ProviderError(f"HTTP {status}; retry budget exhausted")
        result.retries += 1
        time.sleep(delay)
    raise AssertionError("unreachable")


def get_json(provider: str, url: str, interval: float, result: SearchResult,
             **kwargs) -> dict:
    response = request(provider, url, interval, result, **kwargs)
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("expected object")
        return data
    except ValueError:
        raise ProviderError("invalid JSON response") from None
    finally:
        response.close()


def require_records(data: dict, key: str) -> list[dict]:
    records = data.get(key)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ProviderError(f"invalid {key} response schema")
    return records


def validate_paper(paper: dict) -> dict:
    """Check fields used by identity/normalization before admitting a page."""
    if not isinstance(paper.get("title"), str) or not paper["title"].strip():
        raise ProviderError("invalid paper title")
    for key in ("abstract", "paperId", "source", "publicationDate"):
        if paper.get(key) is not None and not isinstance(paper[key], str):
            raise ProviderError(f"invalid paper {key}")
    for key in ("year", "citationCount", "influentialCitationCount"):
        if paper.get(key) is not None and (not isinstance(paper[key], int) or paper[key] < 0):
            raise ProviderError(f"invalid paper {key}")
    for key in ("externalIds", "openAccessPdf", "publicationVenue"):
        if paper.get(key) is not None and not isinstance(paper[key], dict):
            raise ProviderError(f"invalid paper {key}")
    for field in ("DOI", "ArXiv", "SSRN"):
        value = (paper.get("externalIds") or {}).get(field)
        if value is not None and not isinstance(value, str):
            raise ProviderError(f"invalid paper {field}")
    url = (paper.get("openAccessPdf") or {}).get("url")
    if url is not None and not isinstance(url, str):
        raise ProviderError("invalid PDF URL")
    authors = paper.get("authors") or []
    if not isinstance(authors, list) or any(not isinstance(a, dict) or
            not isinstance(a.get("name", ""), str) for a in authors):
        raise ProviderError("invalid paper authors")
    return paper
