"""
Citation task checker for retrieval/citation tasks.

Implements trace-based source existence checks and citation_fidelity per Guide §5.5.
When RAG/tool layer logs retrieved text snippets, source ID, and span, this checker
verifies that citations in the final answer refer to retrieved sources and that
claimed text matches (via fuzzy/span check when available).
"""

import re
from typing import Any, Dict, List, Set, Tuple


def _parse_citation_refs(text: str) -> List[str]:
    """Extract citation references from text, e.g. [1], [2], (Source: X), or [Source: X]."""
    refs: List[str] = []
    # [1], [2], [3] style
    for m in re.finditer(r"\[(\d+)\]", text):
        refs.append(m.group(1))
    # [Source: ...] or (Source: ...) style
    for m in re.finditer(r"(?:\[|\()Source:\s*([^\]\)]+)(?:\]|\))", text, re.IGNORECASE):
        refs.append(m.group(1).strip())
    return refs


def _sources_from_trace(trace: List[Dict[str, Any]]) -> Set[str]:
    """Collect source IDs from TOOL_RESULT events (RAG retrieval results)."""
    sources: Set[str] = set()
    for event in trace:
        if event.get("event_type") != "TOOL_RESULT":
            continue
        payload = event.get("payload") or {}
        result = payload.get("result")
        if isinstance(result, dict):
            sid = result.get("source_id") or result.get("source")
            if sid:
                sources.add(str(sid))
            lst = result.get("results", [])  # Vertex Search format: {"results": [{source_id, text}]}
        elif isinstance(result, list):
            lst = result
        else:
            lst = []
        for item in lst:
            if isinstance(item, dict) and (item.get("source_id") or item.get("source")):
                sources.add(str(item.get("source_id") or item.get("source")))
    return sources


def _snippets_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map source_id -> combined text snippet from TOOL_RESULT events."""
    by_source: Dict[str, List[str]] = {}
    for event in trace:
        if event.get("event_type") != "TOOL_RESULT":
            continue
        payload = event.get("payload") or {}
        result = payload.get("result")
        if isinstance(result, dict):
            sid = str(result.get("source_id") or result.get("source") or "")
            text = result.get("text") or result.get("snippet") or result.get("content") or ""
            if sid:
                by_source.setdefault(sid, []).append(str(text))
            lst = result.get("results", [])
        else:
            lst = result if isinstance(result, list) else []
        for item in lst:
            if isinstance(item, dict):
                sid = str(item.get("source_id") or item.get("source") or "")
                text = item.get("text") or item.get("snippet") or item.get("content") or ""
                if sid:
                    by_source.setdefault(sid, []).append(str(text))
    return {k: " ".join(v) for k, v in by_source.items()}


def _fuzzy_in(snippet: str, pool: str, min_len: int = 20) -> bool:
    """Return True if snippet appears in pool (substring or fuzzy when pool is long)."""
    if not snippet or not pool:
        return False
    snippet = snippet.strip()
    pool = pool.strip()
    if len(snippet) < min_len:
        return snippet in pool
    if snippet in pool:
        return True
    # Relaxed: allow slight wording changes (e.g. leading/trailing words)
    words = set(snippet.lower().split())
    pool_lower = pool.lower()
    return sum(1 for w in words if w in pool_lower) >= max(1, int(0.8 * len(words)))


def check_citation_accuracy(
    final_answer: str,
    gold_answer: Dict[str, Any],
    trace: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Check citation accuracy for retrieval/citation tasks.

    - Source existence: each citation ref in the answer should match a retrieved source in the trace.
    - Claim correctness: text attributed to a source should match retrieved snippets (fuzzy when possible).
    - citation_fidelity = (Correct Citations + Correct Claims) / (Total Citations + Total Claims).
    - pass_fail = 1.0 when citation_fidelity >= 0.8 (per Guide §5.5), else 0.0.

    Args:
        final_answer: Agent's final answer (may contain [1], [Source: X], or inline claims).
        gold_answer: Gold with expected_text_snippet, expected_source_id, expected_source_span.
        trace: Full execution trace (TOOL_RESULT may contain source_id, text/snippet).

    Returns:
        Dictionary with pass_fail, citation_fidelity, and optional breakdown metrics.
    """
    total_citations = 0
    correct_citations = 0
    total_claims = 0
    correct_claims = 0

    refs = _parse_citation_refs(final_answer)
    sources = _sources_from_trace(trace)
    snippets_by_source = _snippets_from_trace(trace)

    expected_snippet = (gold_answer.get("expected_text_snippet") or "").strip()
    expected_source_id = str(gold_answer.get("expected_source_id") or "").strip()

    # Treat “claim” as: the gold expected snippet (or equivalent) appears in the answer
    if expected_snippet:
        total_claims = 1
        if _fuzzy_in(expected_snippet, final_answer, min_len=10):
            correct_claims = 1
        elif expected_snippet in final_answer:
            correct_claims = 1

    # Citation refs in the answer; optional span check when trace has snippets (Guide §5.5)
    if refs:
        total_citations = len(refs)
        for r in refs:
            ok = r in sources
            if not ok and expected_source_id and (r == expected_source_id or expected_source_id.lower() in str(r).lower()):
                ok = True
            if ok and snippets_by_source:
                # Span check: if we have a snippet for this source, treat as extra validation
                snippet = snippets_by_source.get(r) or snippets_by_source.get(expected_source_id)
                if snippet and expected_snippet and not _fuzzy_in(expected_snippet, snippet, min_len=5):
                    ok = False
            if ok:
                correct_citations += 1
    else:
        # No explicit refs: count one “citation” if the answer contains the expected source
        if expected_source_id and expected_source_id in final_answer:
            total_citations = 1
            correct_citations = 1
        elif expected_snippet and correct_claims:
            total_citations = 1
            correct_citations = 1 if (expected_source_id in final_answer or not expected_source_id) else 0

    total = total_citations + total_claims
    correct = correct_citations + correct_claims
    citation_fidelity = (correct / total) if total else 1.0

    # Guide §5.5: hallucination when fidelity < 80%
    pass_fail = 1.0 if citation_fidelity >= 0.8 else 0.0

    # If no refs and no gold claim, use snippet match as primary signal
    if total == 0 and expected_snippet:
        citation_fidelity = 1.0 if correct_claims else 0.0
        pass_fail = 1.0 if correct_claims else 0.0

    return {
        "pass_fail": pass_fail,
        "citation_fidelity": citation_fidelity,
        "correct_citations": float(correct_citations),
        "total_citations": float(total_citations),
        "correct_claims": float(correct_claims),
        "total_claims": float(total_claims),
    }
