"""
rag_tools.py - Local Grounded RAG Retrieval over Geotechnical SOPs and Case Archive.

Plain Python functions for Geotech Advisor Agent.
Operates deterministically without external vector embedding API dependencies.

Grounding rules:
1. Level -> passage resolution is an explicit table lookup (LEVEL_PASSAGE), not a
   similarity search. "RED" must cite TARP-SOP-04, never an incident narrative.
2. Document loading order is sorted, so results are reproducible across machines.
3. Queries that are too vague to ground (< 3 tokens and no explicit passage ID)
   fail closed instead of returning an arbitrary first match.
"""

import os
import re
from typing import Any, Dict, List, Tuple

# Deterministic level -> signed passage mapping (docs/tarp_sop_v1.md).
LEVEL_PASSAGE: Dict[str, str] = {
    "GREEN": "[TARP-SOP-01]",
    "YELLOW": "[TARP-SOP-02]",
    "ORANGE": "[TARP-SOP-03]",
    "RED": "[TARP-SOP-04]",
}

# SOP documents are preferred over case archives when scores tie.
_DOC_PRIORITY = {"tarp_sop_v1.md": 0, "geotech_guidelines.md": 1, "historical_incidents.md": 2}

_PASSAGE_ID_RE = re.compile(r"\[[A-Z0-9_\-]+\]")
_NOT_FOUND = "Information not found in geotechnical documentation."


def _tokenize(text: str) -> set[str]:
    """DRY tokenization helper extracting normalized lowercase words."""
    return set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower()))


def _load_passages(docs_dir: str) -> List[Tuple[str, str, str]]:
    """
    Loads all passages marked with [PASSAGE-ID] from markdown files in docs_dir.
    Returns list of (file_name, passage_id, passage_text) in deterministic order.

    Intentionally uncached: SOP edits must be visible without a process restart.
    """
    passages: List[Tuple[str, str, str]] = []
    if not os.path.isdir(docs_dir):
        return passages

    for fname in sorted(os.listdir(docs_dir), key=lambda n: (_DOC_PRIORITY.get(n, 99), n)):
        if not fname.endswith(".md"):
            continue
        with open(os.path.join(docs_dir, fname), "r", encoding="utf-8") as handle:
            content = handle.read()

        # Split by markdown headers or bracketed IDs [XXX-YYY-NN]
        for block in re.split(r"\n(?=\[[A-Z0-9_\-]+\])", content):
            match = re.search(r"(\[[A-Z0-9_\-]+\])", block)
            if match:
                passages.append((fname, match.group(1), block.strip()))
    return passages


def get_passage(passage_id: str, docs_dir: str = "docs") -> Dict[str, Any]:
    """Exact, deterministic lookup of a signed passage by citation ID."""
    if not passage_id:
        return {"passage_id": None, "text": _NOT_FOUND, "found": False,
                "score": 0.0, "source_file": None}
    for fname, pid, text in _load_passages(docs_dir):
        if pid == passage_id:
            return {"passage_id": pid, "text": text, "found": True,
                    "score": 1.0, "source_file": fname}
    return {"passage_id": None, "text": _NOT_FOUND, "found": False,
            "score": 0.0, "source_file": None}


def get_passage_for_level(level: str, docs_dir: str = "docs") -> Dict[str, Any]:
    """Resolves an operational TARP level to its signed SOP passage (fail safe to Red)."""
    passage_id = LEVEL_PASSAGE.get(str(level).upper().strip(), "[TARP-SOP-04]")
    return get_passage(passage_id, docs_dir=docs_dir)


def retrieve_sop(query: str, docs_dir: str = "docs", min_score: float = 0.15) -> Dict[str, Any]:
    """
    Retrieves grounded SOP passages matching the query.
    Fails closed to 'Information not found' if no passage matches with sufficient relevance.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {"passage_id": None, "text": _NOT_FOUND, "found": False, "score": 0.0,
                "source_file": None}

    # A single bare keyword (e.g. "RED", "rainfall") cannot ground a citation.
    if len(query_tokens) < 3 and not _PASSAGE_ID_RE.search(query):
        return {
            "passage_id": None,
            "text": "Query too vague to ground a citation; supply a passage ID or a full question.",
            "found": False,
            "score": 0.0,
            "source_file": None,
        }

    best: Tuple[str, str, str] = ("", "", "")
    best_score = 0.0

    for fname, pid, text in _load_passages(docs_dir):
        doc_tokens = _tokenize(text)
        if not doc_tokens:
            continue
        # Jaccard / token overlap relevance
        score = len(query_tokens.intersection(doc_tokens)) / float(len(query_tokens))
        if score > best_score:
            best_score = score
            best = (fname, pid, text)

    if best_score >= min_score and best[1]:
        return {
            "passage_id": best[1],
            "text": best[2],
            "found": True,
            "score": round(best_score, 3),
            "source_file": best[0],
        }

    return {
        "passage_id": None,
        "text": _NOT_FOUND,
        "found": False,
        "score": round(best_score, 3),
        "source_file": None,
    }


def find_similar_incidents(
    incident_keywords: List[str],
    docs_dir: str = "docs",
) -> List[Dict[str, Any]]:
    """
    Searches the historical incident archive for past events matching target keywords.
    Returns verified case summaries with citations.
    """
    kw_tokens: set[str] = set()
    for keyword in incident_keywords or []:
        kw_tokens.update(_tokenize(str(keyword)))
    if not kw_tokens:
        return []

    matches: List[Dict[str, Any]] = []
    for fname, pid, text in _load_passages(docs_dir):
        if "INC-" not in pid:
            continue
        overlap = kw_tokens.intersection(_tokenize(text))
        if overlap:
            matches.append({
                "incident_id": pid,
                "matched_keywords": sorted(overlap),
                "summary": text[:200] + "...",
                "full_passage": text,
                "source_file": fname,
            })

    return matches
