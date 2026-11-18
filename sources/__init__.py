"""
sources — paper search adapters.

Every adapter's search_*() function returns a SearchResult, whose records
are dicts in the shared
"raw" schema (mirroring Semantic Scholar's native shape) so a single
parse_paper() in paper_scout.py can normalize all three sources unchanged:

    {
        "title": str, "abstract": str, "year": int | None,
        "citationCount": int, "influentialCitationCount": int,
        "publicationDate": str, "externalIds": {"DOI": ..., "ArXiv": ..., "SSRN": ...},
        "openAccessPdf": {"url": str}, "publicationVenue": {"name": str},
        "authors": [{"name": str}, ...], "paperId": str, "source": str,
    }
"""
from .semantic_scholar import search_semantic_scholar
from .arxiv import search_arxiv
from .arxiv_oai import search_arxiv_index
from .openalex import search_openalex

SEARCH_FUNCS = {
    "semantic_scholar": search_semantic_scholar,
    "arxiv": search_arxiv_index,
    "openalex": search_openalex,
    "arxiv_api": search_arxiv,
}

__all__ = ["search_semantic_scholar", "search_arxiv", "search_arxiv_index",
           "search_openalex", "SEARCH_FUNCS"]
