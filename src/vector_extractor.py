"""
vector_extractor.py - Slab project  (ENGINE A)

Deterministic slab-schedule extraction for VECTOR PDFs (AutoCAD exports / plots).

Such PDFs carry real text + line objects with exact coordinates, so we do NOT
need vision at all: we find the schedule by its title, group the words into rows
by their y-coordinate, and pull the slab fields out of each row with patterns.
This is instant, free, and cannot hallucinate (every value comes from the PDF).

Falls back to "not vector" so the caller can route scanned sheets to the vision
sliding-window engine instead.

Public API:
    is_vector_page(page)              -> bool
    pdf_is_vector(pdf_path)           -> bool
    extract_pdf(pdf_path)             -> list[ {title, region, slabs:[...]} ]
"""

import re

import fitz  # PyMuPDF


# A page with this many extractable words is treated as a real vector drawing
# rather than a scanned image (which yields ~0 words).
_MIN_WORDS_FOR_VECTOR = 40

# Titles that introduce a slab schedule table.
_TITLE_RE = re.compile(r"SCHEDULE\s+OF\s+SLAB|SLAB\s+SCHEDULE", re.I)

# A slab mark: starts with S, mostly short, must look like a label (has a digit
# or is a 2-char code), and is not a header word.
_SLAB_ID_RE = re.compile(r"^S[A-Z]*\d+[A-Z]*$|^S[A-Z]$", re.I)
_HEADER_WORDS = {
    "SLAB", "SHORT", "SPAN", "STEEL", "SUPPORT", "SCHEDULE", "SECTION",
    "STIRRUPS", "SLEEVE", "STAIR", "SB", "SB1", "SB2", "SB3",
}


# ===============================================================
# ROUTER
# ===============================================================

def is_vector_page(page):
    """True if the page has enough real text to be treated as a vector drawing."""
    try:
        return len(page.get_text("words")) >= _MIN_WORDS_FOR_VECTOR
    except Exception:
        return False


def pdf_is_vector(pdf_path):
    """True if the FIRST page is a vector drawing (these sheets are single-page)."""
    try:
        doc = fitz.open(pdf_path)
        return is_vector_page(doc[0])
    except Exception:
        return False


# ===============================================================
# LOCATION
# ===============================================================

def locate_slab_schedules(page):
    """
    Find slab-schedule titles on the page. Returns a list of dicts:
        {"title": str, "anchor": (x, y), "band": fitz.Rect}
    `band` is a generous region below/around the title that should contain the
    whole table; row filtering (below) discards anything that isn't a slab row.
    """
    W, H = page.rect.width, page.rect.height
    words = page.get_text("words")  # (x0,y0,x1,y1, text, block, line, wordno)

    # reconstruct title phrases by scanning text; use search_for for robustness
    titles = []
    for m in _TITLE_RE.finditer(page.get_text("text") or ""):
        pass  # text order is unreliable on CAD sheets; use search_for instead

    hits = page.search_for("SCHEDULE OF SLAB") or page.search_for("SLAB SCHEDULE")
    for r in hits:
        band = fitz.Rect(
            max(0, r.x0 - 60),
            max(0, r.y0 - 6),
            min(W, r.x0 + 470),
            min(H, r.y0 + 0.12 * H),
        )
        titles.append({"title": "SCHEDULE OF SLAB", "anchor": (r.x0, r.y0), "band": band})
    return titles


# ===============================================================
# ROW RECONSTRUCTION + PARSING
# ===============================================================

def _rows_in_band(page, band, y_tol=3.5):
    """Group words inside `band` into visual rows (sorted top-to-bottom)."""
    words = [w for w in page.get_text("words")
             if band.x0 <= w[0] <= band.x1 and band.y0 <= w[1] <= band.y1]
    words.sort(key=lambda w: (w[1], w[0]))
    rows = []
    cur = []
    last_y = None
    for w in words:
        if last_y is None or abs(w[1] - last_y) <= y_tol:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
        last_y = w[1]
    if cur:
        rows.append(cur)
    return [" ".join(t[4] for t in sorted(r, key=lambda w: w[0])) for r in rows]


def _first_slab_id(tokens):
    """Return (slab_id, index) for the first token that is a real slab mark."""
    for i, tok in enumerate(tokens[:4]):
        up = tok.upper().strip(".:,")
        if up in _HEADER_WORDS:
            continue
        if _SLAB_ID_RE.match(up):
            return up, i
    return None, None


def _parse_reinforcement(text):
    """
    Pull bar dia + spacing from a row. Handles '10T @ 150 c/c' and 'T10 @ 150 c/c'
    and 'Y10@150', '#10 @ 150'. Returns (dia_list, spacing_list).
    """
    dia = set()
    spacing = set()

    # dia-first: T10 / Y10 / #10  (optionally followed by @ spacing)
    for m in re.finditer(r"\b([TY#])\s*(\d{1,2})\s*(?:@\s*(\d{2,4})\s*c\s*/?\s*c)?",
                         text, re.I):
        dia.add(f"T{m.group(2)}")
        if m.group(3):
            spacing.add(f"{m.group(3)} C/C")

    # size-first: 10T @ 150 c/c
    for m in re.finditer(r"\b(\d{1,2})\s*[TY#]\s*(?:@\s*(\d{2,4})\s*c\s*/?\s*c)?",
                         text, re.I):
        dia.add(f"T{m.group(1)}")
        if m.group(2):
            spacing.add(f"{m.group(2)} C/C")

    # any remaining bare '@ 150 c/c'
    for m in re.finditer(r"@\s*(\d{2,4})\s*c\s*/?\s*c", text, re.I):
        spacing.add(f"{m.group(1)} C/C")

    return sorted(dia), sorted(spacing)


def parse_slab_row(text):
    """
    Convert one row's text into a slab record, or None if it isn't a slab row.
    Output matches the schema the vision extractors produce.
    """
    tokens = text.split()
    if not tokens:
        return None
    sid, idx = _first_slab_id(tokens)
    if not sid:
        return None

    # Drop any left-bleed from neighbouring drawing text that shares this row
    # (e.g. "3-12T TH. S2 150 ..."): parse only from the slab id onward.
    text = " ".join(tokens[idx:])

    thk = re.search(r"(\d{2,4})\s*(?:mm\s*)?thk", text, re.I)
    if not thk:
        thk = re.search(r"\bthk\.?\s*(\d{2,4})\b", text, re.I)
    mix = re.search(r"\bM\s?(\d{2,3})\b", text)
    typ = re.search(r"\b(ONE|TWO|TOW)\s*WAY\b", text, re.I)

    dia, spacing = _parse_reinforcement(text)

    return {
        "slab_id": sid,
        "thickness": int(thk.group(1)) if thk else None,
        "type": (typ.group(0).upper().replace("TOW", "TWO") if typ else ""),
        "mix": (f"M{mix.group(1)}" if mix else ""),
        "reinforcement": {"dia": dia, "spacing": spacing},
    }


# ===============================================================
# TOP-LEVEL
# ===============================================================

def extract_pdf(pdf_path):
    """
    Extract every slab schedule from a vector PDF.
    Returns a list of {title, region(normalized), slabs:[...]} - one per schedule
    table found. Empty list if none.
    """
    doc = fitz.open(pdf_path)
    results = []

    for page in doc:
        W, H = page.rect.width, page.rect.height
        for sched in locate_slab_schedules(page):
            band = sched["band"]
            rows = _rows_in_band(page, band)
            slabs = [s for s in (parse_slab_row(t) for t in rows) if s]
            if not slabs:
                continue
            # de-dup by slab_id, keep first occurrence order
            seen = set()
            uniq = []
            for s in slabs:
                if s["slab_id"] in seen:
                    continue
                seen.add(s["slab_id"])
                uniq.append(s)
            results.append({
                "title": sched["title"],
                "region": {
                    "x1": round(band.x0 / W, 4), "y1": round(band.y0 / H, 4),
                    "x2": round(band.x1 / W, 4), "y2": round(band.y1 / H, 4),
                },
                "slabs": uniq,
            })
    return results
