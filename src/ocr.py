"""
ocr.py - Slab project

Text for a table region, to give the vision model a SECOND (text) signal next to
the image during pattern detection. Faint cells the model might misread from
pixels are corroborated by recognised text.

Two sources, best first:
  1. Exact PDF vector text inside the region (free, perfect) - for AutoCAD plots
     whose text is real characters.
  2. Tesseract OCR on the (already contrast-enhanced) crop - for outline-text or
     scanned sheets where there is no vector text.

Everything degrades gracefully: if Tesseract isn't installed, region_text just
returns "" and the caller falls back to image-only detection (no regression).
"""

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except Exception:
    _OCR_OK = False


def _clean(text):
    """Drop blank and mostly-symbol garbage lines (CAD linework OCRs to noise)."""
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        alnum = sum(c.isalnum() for c in ln)
        if alnum >= 3 and alnum / max(len(ln), 1) > 0.4:
            out.append(ln)
    return "\n".join(out)


def vector_text_in_region(pdf_path, region, page_index=0, min_chars=20):
    """
    Exact PDF text whose word-centres fall inside the normalized region, returned
    in reading order (rows by y, words by x). Returns '' if the region has little
    real text (e.g. outline-curve or scanned sheets).
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        W, H = page.rect.width, page.rect.height
        x1, y1 = region["x1"] * W, region["y1"] * H
        x2, y2 = region["x2"] * W, region["y2"] * H

        rows = {}
        for w in page.get_text("words"):
            cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                rows.setdefault(round(cy / 3), []).append(w)

        lines = []
        for key in sorted(rows):
            words = sorted(rows[key], key=lambda w: w[0])
            lines.append(" ".join(t[4] for t in words))
        text = "\n".join(lines).strip()

        return text if sum(c.isalnum() for c in text) >= min_chars else ""
    except Exception:
        return ""


def ocr_image(image_path, psm=6):
    """Tesseract OCR on a crop (use the contrast-enhanced one). '' if unavailable."""
    if not _OCR_OK:
        return ""
    try:
        return _clean(pytesseract.image_to_string(
            Image.open(image_path), config=f"--psm {psm}"))
    except Exception:
        return ""


def region_text(pdf_path, region, crop_png=None, page_index=0):
    """
    Best available text for a region.
    Returns (text, source) where source is 'vector', 'ocr', or 'none'.
    """
    text = vector_text_in_region(pdf_path, region, page_index=page_index)
    if text:
        return text, "vector"
    if crop_png:
        ocr = ocr_image(crop_png)
        if ocr:
            return ocr, "ocr"
    return "", "none"
