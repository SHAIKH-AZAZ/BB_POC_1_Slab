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
# A slab-schedule title is any heading line that mentions a SLAB together with a
# SCHEDULE/DETAILS/REINFORCEMENT word, in either order, with anything in between
# (RCC, R.C.C., TWO-WAY, OF, etc.). Far more robust than an exact-string list.
_SLAB_WORD = r"\bSLABS?\b"
_SCHED_WORD = r"\b(?:SCHEDULE|DETAILS|REINF(?:ORCEMENT)?)\b"
_TITLE_RE = re.compile(
    rf"(?:{_SLAB_WORD}.*{_SCHED_WORD}|{_SCHED_WORD}.*{_SLAB_WORD})", re.I
)
# Don't mistake a beam/column/footing schedule line for a slab one.
_OTHER_FAMILY_RE = re.compile(r"\b(BEAM|COLUMN|FOOTING|STAIR|LINTEL|PILE)S?\b", re.I)

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
    Find slab-schedule titles on the page by FUZZY matching, not an exact list.

    Words are grouped into visual lines; any line whose text matches _TITLE_RE
    (mentions a SLAB + a SCHEDULE/DETAILS/REINF word, in any order) is treated as
    a title — unless it clearly belongs to another family (beam/column/footing...).
    This catches 'SCHEDULE OF RCC SLABS', 'SLAB REINFORCEMENT SCHEDULE',
    'R.C.C. SLAB DETAILS', 'TWO WAY SLAB SCHEDULE', etc.

    Returns [{"title": str, "anchor": (x, y), "band": fitz.Rect}, ...].
    """
    W, H = page.rect.width, page.rect.height
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,wordno)
    if not words:
        return titles_via_search(page)  # fallback for odd text encodings

    # group words into visual lines by y
    rows = {}
    for w in words:
        rows.setdefault(round(w[1] / 3.0), []).append(w)

    titles = []
    for key in sorted(rows):
        line_words = sorted(rows[key], key=lambda w: w[0])
        text = " ".join(t[4] for t in line_words).strip()
        # Must read like a slab schedule heading. (Combined 'SLAB & BEAM SCHEDULE'
        # is fine — the row parser only accepts S-prefixed slab marks, so beam
        # rows can never be mis-parsed as slabs.) Keep it short to avoid matching
        # a long note sentence that happens to contain both words.
        if len(text) > 60 or not _TITLE_RE.search(text):
            continue
        x0 = min(t[0] for t in line_words)
        y0 = min(t[1] for t in line_words)
        band = fitz.Rect(
            max(0, x0 - 60), max(0, y0 - 6),
            min(W, x0 + 470), min(H, y0 + 0.12 * H),
        )
        titles.append({"title": text[:60], "anchor": (x0, y0), "band": band})
    return titles


def titles_via_search(page):
    """Fallback title finder using search_for for a few common spellings."""
    W, H = page.rect.width, page.rect.height
    out = []
    for needle in ("SCHEDULE OF SLAB", "SLAB SCHEDULE", "SCHEDULE OF RCC SLABS"):
        for r in page.search_for(needle):
            band = fitz.Rect(max(0, r.x0 - 60), max(0, r.y0 - 6),
                             min(W, r.x0 + 470), min(H, r.y0 + 0.12 * H))
            out.append({"title": needle, "anchor": (r.x0, r.y0), "band": band})
    return out


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


def _count_slab_ids(text):
    """Count slab-mark tokens in one row. >=2 on a single row => transposed table."""
    n = 0
    for tok in text.split():
        up = tok.upper().strip(".:,")
        if up in _HEADER_WORDS:
            continue
        if _SLAB_ID_RE.match(up):
            n += 1
    return n


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
# TRANSPOSED TABLES (slab marks are COLUMN headers, not rows)
# ===============================================================

def _word_rows_in_band(page, band, y_tol=3.5):
    """Group words inside `band` into visual rows; return a list of word-lists."""
    words = [w for w in page.get_text("words")
             if band.x0 <= w[0] <= band.x1 and band.y0 <= w[1] <= band.y1]
    words.sort(key=lambda w: (w[1], w[0]))
    rows, cur, last_y = [], [], None
    for w in words:
        if last_y is None or abs(w[1] - last_y) <= y_tol:
            cur.append(w)
        else:
            rows.append(cur); cur = [w]
        last_y = w[1]
    if cur:
        rows.append(cur)
    return rows


def _slab_marks_in_row(word_row):
    """Return [(mark, x_center, word)] for the slab-mark words in one row."""
    out = []
    for w in word_row:
        up = w[4].upper().strip(".:,")
        if up in _HEADER_WORDS:
            continue
        if _SLAB_ID_RE.match(up):
            out.append((up, (w[0] + w[2]) / 2.0, w))
    return out


def _parse_transposed_blob(slab_id, tokens):
    """Parse one column's stacked value tokens (top-to-bottom) into a slab record."""
    text = " ".join(tokens)
    mix = re.search(r"\bM\s?(\d{2,3})\b", text)
    typ = re.search(r"\b(ONE|TWO|TOW)\s*WAY\b", text, re.I)
    dia, spacing = _parse_reinforcement(text)
    # thickness: first 2-4 digit number in slab range that isn't a bar spacing
    thickness = None
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d{2,4}", tok):
            v = int(tok)
            prev = tokens[i - 1] if i else ""
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if 75 <= v <= 400 and prev != "@" and nxt.lower() not in ("c/c", "cc"):
                thickness = v
                break
    return {
        "slab_id": slab_id,
        "thickness": thickness,
        "type": (typ.group(0).upper().replace("TOW", "TWO") if typ else ""),
        "mix": (f"M{mix.group(1)}" if mix else ""),
        "reinforcement": {"dia": dia, "spacing": spacing},
    }


def _parse_transposed(page, header_words, page_h):
    """
    Read a TRANSPOSED schedule: slab marks are column headers and each property
    (THK, MIX, TYPE, reinforcement) runs DOWN its column. Every value word is
    assigned to its nearest slab column by x; each column becomes one slab.
    Row labels (to the right of the last column) are excluded.
    """
    marks = _slab_marks_in_row(header_words)
    if len(marks) < 2:
        return []
    marks.sort(key=lambda m: m[1])
    xs = [m[1] for m in marks]
    gap = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 1.0
    label_x = xs[-1] + gap * 0.5             # labels sit just right of the last column
    header_bottom = max(m[2][3] for m in marks)
    y_limit = header_bottom + 0.22 * page_h  # generous table height below the header

    cols = {i: [] for i in range(len(marks))}
    for w in page.get_text("words"):
        cx = (w[0] + w[2]) / 2.0
        cy = (w[1] + w[3]) / 2.0
        if cx >= label_x or cy <= header_bottom or cy > y_limit:
            continue
        if not (xs[0] - gap <= cx <= xs[-1] + gap * 0.4):
            continue
        i = min(range(len(xs)), key=lambda j: abs(cx - xs[j]))
        if abs(cx - xs[i]) <= gap * 0.6:
            cols[i].append(w)

    slabs = []
    for i, (mark, _, _) in enumerate(marks):
        toks = [w[4] for w in sorted(cols[i], key=lambda w: (w[1], w[0]))]
        slabs.append(_parse_transposed_blob(mark, toks))
    return slabs


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
            word_rows = _word_rows_in_band(page, band)

            # A row carrying several slab marks (S1 S2 S3 S4 ...) means the slabs
            # are COLUMN headers -> read the table the transposed way (down each
            # column). Otherwise read it the normal way (one slab per row).
            header = next(
                (r for r in word_rows if len(_slab_marks_in_row(r)) >= 2), None
            )
            if header is not None:
                slabs = _parse_transposed(page, header, H)
            else:
                rows = [" ".join(t[4] for t in sorted(r, key=lambda w: w[0]))
                        for r in word_rows]
                slabs = [parse_slab_row(t) for t in rows]

            slabs = [s for s in slabs if s]
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
