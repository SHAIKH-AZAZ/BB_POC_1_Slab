"""
Deterministic post-extraction recovery for slab reinforcement.

Many slab PDFs have a clean text layer that pdfplumber can extract row by row.
The vision model occasionally drops spacing (e.g. emits dia=['T12'] but no
spacing for a cell that clearly shows 'T12 @ 100 C/C').  This module re-reads
the PDF, locates each slab row, and fills any missing reinforcement values
from the source text.

It is a pure backstop: it never overwrites correct model output, only fills
gaps, and it is a no-op for image-only PDFs (returns empty results so the
caller can warn).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

try:
    import pdfplumber
except ImportError:  # pragma: no cover - dependency listed in requirements.txt
    pdfplumber = None

from extraction_guard import extract_rebar_parts


_SLAB_ID_AT_START = re.compile(r"^([A-Z]{1,3}\d{0,3}[A-Z]?)\s+")


def _read_pdf_lines(pdf_path: str) -> List[str]:
    """Return all non-empty stripped lines from every page (text layer only)."""
    if pdfplumber is None or not os.path.exists(pdf_path):
        return []
    lines: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        lines.append(line)
    except Exception:
        return []
    return lines


def _index_lines_by_slab_id(lines: List[str]) -> Dict[str, List[str]]:
    """
    Group lines under the slab_id token they begin with.  Continuation lines
    (no slab_id at the start) attach to the previous slab.
    """
    grouped: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in lines:
        match = _SLAB_ID_AT_START.match(line)
        if match:
            current = match.group(1)
            grouped.setdefault(current, []).append(line)
        elif current is not None:
            grouped.setdefault(current, []).append(line)
    return grouped


def _merge_unique(existing: List[str], extras: List[str]) -> List[str]:
    out = list(existing)
    for value in extras:
        if value and value not in out:
            out.append(value)
    return out


def recover_slabs_from_pdf(pdf_path: str, slabs: List[dict]) -> Dict[str, dict]:
    """
    For every slab record, look up its slab_id in the PDF text and return
    reinforcement (dia + spacing) parsed deterministically from the text.

    Returns a dict keyed by slab_id with the parsed reinforcement.  Missing
    slab_ids and image-only PDFs simply return an empty dict.
    """
    lines = _read_pdf_lines(pdf_path)
    if not lines:
        return {}
    grouped = _index_lines_by_slab_id(lines)
    recovered: Dict[str, dict] = {}
    for slab in slabs:
        slab_id = str(slab.get("slab_id") or "").strip()
        if not slab_id or slab_id not in grouped:
            continue
        joined = " ".join(grouped[slab_id])
        parts = extract_rebar_parts(joined)
        if parts["dia"] or parts["spacing"]:
            recovered[slab_id] = parts
    return recovered


def merge_recovered(slab: dict, recovered: dict) -> dict:
    """
    Fill any missing reinforcement values on `slab` using `recovered` without
    overwriting existing model-supplied data.  Returns the slab in place.
    """
    if not recovered:
        return slab
    reinf = slab.setdefault("reinforcement", {"dia": [], "spacing": []})
    reinf["dia"] = _merge_unique(reinf.get("dia") or [], recovered.get("dia") or [])
    reinf["spacing"] = _merge_unique(
        reinf.get("spacing") or [], recovered.get("spacing") or []
    )
    return slab


def recover_and_merge(pdf_path: str, slabs: List[dict]) -> List[dict]:
    """Convenience: recover from PDF text and merge into each slab in place."""
    recovered_map = recover_slabs_from_pdf(pdf_path, slabs)
    for slab in slabs:
        slab_id = str(slab.get("slab_id") or "").strip()
        if slab_id and slab_id in recovered_map:
            merge_recovered(slab, recovered_map[slab_id])
    return slabs
