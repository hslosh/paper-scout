"""Conservative paper identity matching and lossless retrieval provenance."""
from copy import deepcopy
import hashlib
import re
import unicodedata


def normalize_doi(value: str) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()


def normalize_arxiv(value: str) -> str:
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value.strip(), flags=re.I)
    return re.sub(r"v\d+$", "", value.removesuffix(".pdf"))


def _text(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def aliases(raw: dict) -> set[str]:
    ids = raw.get("externalIds") or {}
    keys = set()
    for field, prefix, normalize in (("DOI", "doi", normalize_doi),
                                     ("ArXiv", "arxiv", normalize_arxiv),
                                     ("SSRN", "ssrn", str)):
        if ids.get(field):
            keys.add(f"{prefix}:{normalize(str(ids[field]))}")
    if raw.get("paperId"):
        keys.add(f"{raw.get('source', 'unknown')}:{raw['paperId']}")
    return keys


def _title_key(raw: dict) -> tuple:
    authors = tuple(sorted(_text(a.get("name", "")) for a in (raw.get("authors") or []) if a.get("name")))
    return _text(raw.get("title", "")), raw.get("year"), authors


def _doi(raw: dict) -> str:
    return normalize_doi(str((raw.get("externalIds") or {}).get("DOI") or ""))


def merge_records(records: list[dict]) -> list[dict]:
    # Indexed by alias and complete title key so merging stays linear-ish; a
    # merged group moves to the end, matching the original list-append order.
    groups: dict[int, list[dict]] = {}
    group_dois: dict[int, set[str]] = {}
    by_alias: dict[str, int] = {}
    by_title: dict[tuple, set[int]] = {}
    for gid, raw in enumerate(records):
        key = _title_key(raw)
        doi = _doi(raw)
        matching = {by_alias[a] for a in aliases(raw) if a in by_alias}
        if key[0] and key[1] and key[2]:
            matching |= {g for g in by_title.get(key, ())
                         if not (doi and group_dois[g] and doi not in group_dois[g])}
        merged = [raw]
        for g in sorted(matching, reverse=True):
            merged.extend(groups.pop(g))
            group_dois.pop(g)
        groups[gid] = merged
        group_dois[gid] = {_doi(r) for r in merged} - {""}
        for r in merged:
            for alias in aliases(r):
                by_alias[alias] = gid
            r_key = _title_key(r)
            if r_key[0] and r_key[1] and r_key[2]:
                by_title.setdefault(r_key, set()).difference_update(matching)
                by_title[r_key].add(gid)

    output = []
    for group in groups.values():
        # Metadata richness, not provider ordering, selects the representative.
        ordered = sorted(group, key=lambda r: len(r.get("abstract") or ""), reverse=True)
        best = deepcopy(ordered[0])
        for r in ordered[1:]:
            for key, value in r.items():
                if not best.get(key) and value:
                    best[key] = deepcopy(value)
        ids = {}
        for r in reversed(ordered):
            ids.update({k: v for k, v in (r.get("externalIds") or {}).items() if v})
        if ids.get("DOI"):
            ids["DOI"] = normalize_doi(ids["DOI"])
        best["externalIds"] = ids
        best["provenance"] = deepcopy(group)
        best["sources"] = sorted({r.get("source", "unknown") for r in group})
        best["queries"] = sorted({r.get("query", "") for r in group})
        best["oa_pdfs"] = sorted({(r.get("openAccessPdf") or {}).get("url") for r in group} - {None, ""})
        output.append(best)
    return output


def paper_key(paper: dict) -> str:
    if paper.get("doi"):
        return "doi:" + normalize_doi(paper["doi"])
    if paper.get("arxiv_id"):
        return "arxiv:" + normalize_arxiv(paper["arxiv_id"])
    if paper.get("ssrn_id"):
        return "ssrn:" + str(paper["ssrn_id"])
    if paper.get("paper_id"):
        return f"{paper.get('source', 'unknown')}:{paper['paper_id']}"
    return f"title:{_text(paper['title'])}|{paper.get('year', '')}|{_text(paper.get('authors', ''))}"


def file_id(paper: dict) -> str:
    return hashlib.sha256(paper_key(paper).encode("utf-8")).hexdigest()[:24]
