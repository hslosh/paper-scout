"""Retrieve provider-declared OA PDFs, with Unpaywall as a fallback."""
import hashlib
import os
from pathlib import Path
import tempfile
import time

import pypdf
import requests

import config
from identity import file_id, normalize_doi

MAX_PDF_BYTES = 25 * 1024 * 1024


def cache_file_id(paper: dict) -> str:
    # Versioned locations must not reuse an older preprint's cached contents.
    locations = sorted(set(filter(None, [paper.get("oa_pdf", ""), *paper.get("oa_pdfs", [])])))
    version = repr((paper.get("arxiv_id", ""), locations))
    return file_id(paper) + "-" + hashlib.sha256(version.encode("utf-8")).hexdigest()[:12]


def _valid_pdf(path: Path) -> bool:
    try:
        if not 5 <= path.stat().st_size <= MAX_PDF_BYTES:
            return False
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                return False
            stream.seek(0)
            return len(pypdf.PdfReader(stream).pages) > 0
    except Exception:
        return False


def _download(url: str, dest: Path) -> str:
    if not url.startswith(("https://", "http://")):
        return "unsupported_url"
    temp = None
    try:
        with requests.get(url, timeout=(10, 30), stream=True,
                          headers={"User-Agent": "paper-scout/1.1 (academic use)"}) as response:
            response.raise_for_status()
            if int(response.headers.get("Content-Length") or 0) > MAX_PDF_BYTES:
                return "too_large"
            with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as handle:
                temp = Path(handle.name)
                size = 0
                deadline = time.monotonic() + 60
                for chunk in response.iter_content(chunk_size=65536):
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        return "too_large"
                    if time.monotonic() > deadline:
                        return "download_timeout"
                    handle.write(chunk)
        if not _valid_pdf(temp):
            return "invalid_pdf"
        os.replace(temp, dest)
        return "downloaded"
    except (requests.RequestException, ValueError, OSError) as exc:
        return type(exc).__name__
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def fetch_fulltext_pdf(paper: dict, pdf_dir: Path | None = None) -> str:
    """Return a checked cached/downloaded PDF; record status and safe diagnostics."""
    pdf_dir = Path(pdf_dir) if pdf_dir is not None else config.PDF_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / f"{cache_file_id(paper)}.pdf"
    paper["fulltext_attempts"] = []
    if dest.exists() and _valid_pdf(dest):
        paper["fulltext_status"] = "cached"
        paper["pdf_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
        return str(dest)

    def attempt(url):
        status = _download(url, dest)
        paper["fulltext_attempts"].append({"url": url, "status": status})
        if status == "downloaded":
            paper["fulltext_status"] = status
            paper["pdf_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            return str(dest)
        return ""

    urls = list(dict.fromkeys([paper.get("oa_pdf", ""), *paper.get("oa_pdfs", [])]))
    for url in filter(None, urls):
        path = attempt(url)
        if path:
            return path

    doi = normalize_doi(paper.get("doi") or "")
    if doi and config.UNPAYWALL_EMAIL:
        try:
            with requests.get(f"{config.UNPAYWALL_API_BASE}/{doi}",
                              params={"email": config.UNPAYWALL_EMAIL}, timeout=15) as response:
                response.raise_for_status()
                data = response.json()
            locations = [data.get("best_oa_location") or {}, *(data.get("oa_locations") or [])]
            for location in locations:
                url = location.get("url_for_pdf")
                if url and url not in urls:
                    urls.append(url)
                    path = attempt(url)
                    if path:
                        return path
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            paper["fulltext_attempts"].append({"provider": "unpaywall", "status": type(exc).__name__})
    paper["fulltext_status"] = "unavailable" if not paper["fulltext_attempts"] else "failed"
    return ""


def extract_pdf_text(pdf_path: str, max_chars: int = 20000) -> str:
    """Extract plain text from a local PDF (best-effort). Returns "" on failure —
    callers should fall back to the abstract, never crash on a malformed PDF."""
    if not pdf_path:
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        chunks = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                break
        return "".join(chunks)[:max_chars]
    except Exception as e:
        print(f"  [WARN] PDF text extraction failed for {pdf_path}: {e}")
        return ""
