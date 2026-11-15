"""
rag_tools.py - Local Grounded RAG Retrieval over Geotechnical SOPs and Case Archive.

Plain Python functions for Geotech Advisor Agent.
Operates deterministically without external vector embedding API dependencies.
"""

import functools
import os
import re
from typing import Any, Dict, List, Optional, Tuple


def _tokenize(text: str) -> set[str]:
    """DRY tokenization helper extracting normalized lowercase words."""
    return set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower()))


@functools.lru_cache(maxsize=16)
def _load_passages(docs_dir: str) -> List[Tuple[str, str, str]]:
    """
    Loads all passages marked with [PASSAGE-ID] from markdown files in docs_dir.
    Returns list of (file_name, passage_id, passage_text).
    """
    passages = []
    if not os.path.exists(docs_dir):
        return passages

    for fname in os.listdir(docs_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by markdown headers or bracketed IDs [XXX-YYY-NN]
            raw_blocks = re.split(r"\n(?=\[[A-Z0-9_\-]+\])", content)
            for block in raw_blocks:
                match = re.search(r"(\[[A-Z0-9_\-]+\])", block)
                if match:
                    passage_id = match.group(1)
                    text = block.strip()
                    passages.append((fname, passage_id, text))
    return passages


def retrieve_sop(query: str, docs_dir: str = "docs", min_score: float = 0.15) -> Dict[str, Any]:
    """
    Retrieves grounded SOP passages matching the query.
    Fails closed to 'Information not found' if no passage matches with sufficient relevance.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {
            "passage_id": None,
            "text": "Information not found in geotechnical documentation.",
            "found": False,
            "score": 0.0,
        }

    passages = _load_passages(docs_dir)
    best_passage = None
    best_id = None
    best_score = 0.0

    for fname, pid, text in passages:
        doc_tokens = _tokenize(text)
        if not doc_tokens:
            continue
        # Jaccard / token overlap relevance
        overlap = query_tokens.intersection(doc_tokens)
        score = len(overlap) / float(len(query_tokens))

        if score > best_score:
            best_score = score
            best_passage = text
            best_id = pid

    if best_score >= min_score and best_id is not None:
        return {
            "passage_id": best_id,
            "text": best_passage,
            "found": True,
            "score": round(best_score, 3),
        }

    return {
        "passage_id": None,
        "text": "Information not found in geotechnical documentation.",
        "found": False,
        "score": round(best_score, 3),
    }


def find_similar_incidents(
    incident_keywords: List[str],
    docs_dir: str = "docs",
) -> List[Dict[str, Any]]:
    """
    Searches the historical incident archive for past events matching target keywords.
    Returns verified case summaries with citations.
    """
    kw_tokens = set(k.lower() for k in incident_keywords)
    passages = _load_passages(docs_dir)
    matches = []

    for fname, pid, text in passages:
        if "INC-" in pid:
            doc_tokens = _tokenize(text)
            overlap = kw_tokens.intersection(doc_tokens)
            if overlap:
                matches.append({
                    "incident_id": pid,
                    "matched_keywords": list(overlap),
                    "summary": text[:200] + "...",
                    "full_passage": text,
                })

    return matches
