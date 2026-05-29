"""
One-shot script: walk every output/<stem>/<stem>.json, recover dia/spacing
from the matching input/<stem>.pdf, and rewrite the JSON in place.

Useful after fixing extraction bugs without re-running the full vision pipeline.
"""

import json
import os
import re
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import INPUT_DIR, OUTPUT_DIR  # noqa: E402
from extraction_guard import _is_valid_spacing  # noqa: E402
from text_recovery import recover_and_merge  # noqa: E402


def _clean_spacings(values):
    """Drop invalid (non-numeric or out-of-range) spacing values."""
    out = []
    for value in values or []:
        text = str(value).strip().upper()
        if not text:
            continue
        m = re.match(r"^(\d{2,4})", text)
        if not m or not _is_valid_spacing(m.group(1)):
            continue
        out.append(f"{m.group(1)} C/C")
    return out


def normalize_slab(slab):
    """Coerce legacy schema variants into the canonical shape."""
    slab_id = slab.get("slab_id") or slab.get("slab_type") or ""
    reinf = slab.get("reinforcement") or {}

    # Legacy pattern-1 also stored along/across span as separate arrays.
    along = reinf.get("along_span") or []
    across = reinf.get("across_span") or []

    canonical = {
        "slab_id": slab_id,
        "thickness": slab.get("thickness"),
        "type": slab.get("type") or "",
        "mix": slab.get("mix") or "",
        "reinforcement": {
            "dia": list(reinf.get("dia") or []),
            "spacing": _clean_spacings(reinf.get("spacing")),
        },
    }
    remarks = slab.get("remarks")
    if remarks not in (None, ""):
        canonical["remarks"] = remarks

    # If the legacy along/across arrays exist, parse them too so we don't
    # discard data that the model genuinely captured.
    if along or across:
        from extraction_guard import extract_rebar_parts

        parts = extract_rebar_parts(*along, *across)
        for value in parts["dia"]:
            if value not in canonical["reinforcement"]["dia"]:
                canonical["reinforcement"]["dia"].append(value)
        for value in parts["spacing"]:
            if value not in canonical["reinforcement"]["spacing"]:
                canonical["reinforcement"]["spacing"].append(value)

    return canonical


def reconcile_one(stem):
    pdf_path = os.path.join(INPUT_DIR, f"{stem}.pdf")
    json_path = os.path.join(OUTPUT_DIR, stem, f"{stem}.json")

    if not os.path.exists(pdf_path):
        print(f"⚠ skipping {stem}: input PDF not found")
        return None
    if not os.path.exists(json_path):
        print(f"⚠ skipping {stem}: output JSON not found")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    slabs = payload.get("slabs") or []
    canonical = [normalize_slab(s) for s in slabs]

    before = sum(
        len(s["reinforcement"]["dia"]) + len(s["reinforcement"]["spacing"])
        for s in canonical
    )
    recover_and_merge(pdf_path, canonical)

    # Sort + dedupe to keep output stable.
    for slab in canonical:
        reinf = slab["reinforcement"]
        reinf["dia"] = sorted(set(reinf["dia"]))
        reinf["spacing"] = sorted(set(reinf["spacing"]))

    after = sum(
        len(s["reinforcement"]["dia"]) + len(s["reinforcement"]["spacing"])
        for s in canonical
    )

    payload["slabs"] = canonical
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return before, after


def main():
    pdfs = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")
    )
    for pdf in pdfs:
        stem = os.path.splitext(pdf)[0]
        result = reconcile_one(stem)
        if result is None:
            continue
        before, after = result
        delta = after - before
        flag = "✅" if delta > 0 else "•"
        print(f"{flag} {stem}: {before} → {after} reinforcement values (+{delta})")


if __name__ == "__main__":
    main()
