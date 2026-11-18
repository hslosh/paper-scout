"""
vault.py — write Obsidian source notes for kept papers.
"""
import re
import json
from datetime import date
from pathlib import Path

import config
from identity import file_id


def _sanitize_md(text: str) -> str:
    """
    Sanitize untrusted text before embedding in an Obsidian markdown note.
    - Collapses ]] to prevent stray wikilinks
    - Removes leading --- that could open a second frontmatter block
    - Escapes backtick runs that could break code fences
    """
    text = text.replace("]]", "] ]")
    text = re.sub(r"^---", "—", text, flags=re.MULTILINE)
    text = re.sub(r"```", "'''", text)
    return text.strip()


def save_obsidian_note(paper: dict, query: str, hypothesis: str = "") -> Path:
    """
    Create a source note, preserving any existing note unchanged. If `hypothesis` is a non-empty
    draft (not an "[OLLAMA ERROR]"/"[OLLAMA SKIP]" string), it's inserted into
    the "Key Ideas" section; otherwise that section is left blank as before.
    """
    today = date.today().isoformat()
    fname = config.VAULT_PATH / f"paper-{file_id(paper)}.md"
    if fname.exists():
        return fname  # Existing notes (including user edits) are never replaced.

    abstract_safe    = _sanitize_md(paper["abstract"])
    score_notes_safe = _sanitize_md(paper["score_notes"])
    query_safe       = query.replace('"', "'")
    source_safe      = paper.get("source", "semantic_scholar")

    is_real_draft = hypothesis and not hypothesis.startswith(("[OLLAMA ERROR]", "[OLLAMA SKIP]"))
    if is_real_draft:
        key_ideas_section = _sanitize_md(hypothesis)
    elif hypothesis:
        # An error/skip message — keep it visible (so you know why it's blank)
        # but clearly marked as a tool note, not a real idea.
        key_ideas_section = f"- *(hypothesis drafting skipped: {hypothesis})*"
    else:
        key_ideas_section = "-"

    content = f"""---
title: {json.dumps(paper['title'], ensure_ascii=False)}
source: {json.dumps(paper['url'], ensure_ascii=False)}
date: {today}
tags: [trading, paper-scout, research, auto-scouted, {source_safe}]
citations: {paper['citations']}
scout_score: {paper['score']}
status: inbox
---

## Summary
{abstract_safe}

## Key Ideas
{key_ideas_section}

## Score Notes
{score_notes_safe}

## My Thoughts


## Links
- Query: "{query_safe}"  (source: {source_safe})
"""
    config.VAULT_PATH.mkdir(parents=True, exist_ok=True)
    try:
        with fname.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        pass  # Another process created the note while this one was preparing it.
    return fname
