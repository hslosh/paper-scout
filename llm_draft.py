"""
llm_draft.py — local LLM hypothesis drafting via Ollama.

No external API calls, no cost, runs entirely local. If Ollama isn't running
(or the model isn't pulled), draft_hypothesis() returns a message prefixed
with "[OLLAMA ERROR]" or "[OLLAMA SKIP]" instead of raising — callers check
for that prefix and skip gracefully rather than crash a whole scout run over
one missing local service.
"""
import requests

import config

PROMPT_TEMPLATE = """You are a skeptical quantitative trading research assistant. Given the \
paper text below, draft a concise, TESTABLE trading hypothesis. Use exactly these five \
section headers, each 1-4 sentences:

**Core idea** (the edge, and why it plausibly exists)
**Required conditions** (instrument, market regime, data needed)
**Entry trigger** (precise and mechanical - something you could code)
**Exit logic / invalidation criteria**
**Expected win rate, frequency, RR, and why**

Be specific and skeptical. If the paper's evidence is thin, weak, or the edge looks like \
it's just beta/drift in disguise, say so plainly instead of padding with disclaimers.

Title: {title}

Text:
{text}
"""


def draft_hypothesis(paper: dict, model: str = config.OLLAMA_MODEL,
                     full_text: str = "", timeout: int = config.OLLAMA_TIMEOUT_SEC) -> str:
    """
    Call a local Ollama instance to draft a trading hypothesis from a paper's
    full text (if retrieved via fulltext.py) or its abstract otherwise.

    Returns the draft, or a "[OLLAMA ERROR] ..." / "[OLLAMA SKIP] ..." string.
    Callers (see vault.save_obsidian_note) must check for those prefixes and
    degrade gracefully instead of writing garbage into a note.
    """
    # Prefer: PDF full text > untruncated abstract > the 600-char CSV preview.
    text = (full_text or paper.get("abstract_full") or paper.get("abstract") or "").strip()
    if not text:
        return "[OLLAMA SKIP] no abstract or full text available for this paper"

    prompt = PROMPT_TEMPLATE.format(title=paper.get("title", ""), text=text[:8000])

    try:
        resp = requests.post(
            config.OLLAMA_API_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        return (f"[OLLAMA ERROR] could not connect to {config.OLLAMA_API_URL} - "
                f"is Ollama running? Start it with `ollama serve`, and make sure "
                f"the model is pulled: `ollama pull {model}`")
    except requests.exceptions.Timeout:
        return f"[OLLAMA ERROR] request timed out after {timeout}s (model: {model})"
    except Exception as e:
        return f"[OLLAMA ERROR] {e}"

    if resp.status_code == 404:
        return (f"[OLLAMA ERROR] model '{model}' not found - pull it first: "
                f"`ollama pull {model}`")
    try:
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("response", ""), str):
            return "[OLLAMA ERROR] invalid response schema"
    except Exception as e:
        return f"[OLLAMA ERROR] bad response from Ollama: {e}"

    draft = (data.get("response") or "").strip()
    return draft or "[OLLAMA ERROR] empty response from model"
