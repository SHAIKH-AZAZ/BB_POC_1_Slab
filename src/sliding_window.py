"""
sliding_window.py - Slab project  (ENGINE B)

Vision-based slab-schedule localization for SCANNED PDFs (raster images with no
extractable text, so Engine A / pdfplumber can't help).

A full A-size sheet is too large for the vision model to read a small table in
one shot (the API downsamples it). So we tile the high-DPI page into overlapping
windows, ask the model per tile "is there a slab schedule table here?", then merge
the positive tiles into one region per schedule. The caller crops that region and
runs the normal classify -> main_1..9 extraction on it.

Public API:
    locate_slab_schedules(pdf_path, temp_folder, dpi=300) -> list[ {region, confidence} ]
"""

import os

import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from vision_extractor import extract_structured


# Each rendered tile is capped to this many pixels on its long side. Big enough
# for the model to read a small table, small enough to never hit PIL's
# decompression-bomb limit on A0/A1 sheets (which at 400 DPI are ~1e9 pixels).
TILE_LONG_PX = 1600


# ---- tiling parameters ----
TILE_FRAC = 0.40     # each tile is 40% of page width/height ...
STEP_FRAC = 0.25     # ... stepped by 25% -> ~37% overlap so nothing falls in a seam
MIN_TILE_CONF = 0.55


class TileVerdict(BaseModel):
    has_slab_schedule: bool = Field(
        description="True ONLY if this crop shows a slab reinforcement SCHEDULE "
                    "TABLE: rows of slab marks (S1, S2, ...) with thickness / "
                    "reinforcement columns. A plan/layout drawing is NOT a schedule."
    )
    confidence: float = Field(description="0.0 - 1.0")


_TILE_PROMPT = """This is a CROP of a structural drawing sheet.
Does this crop contain a SLAB REINFORCEMENT SCHEDULE TABLE — a gridded table whose
rows are slab marks (S1, S2, S3 ...) with columns like THICKNESS, TYPE and
reinforcement (bar dia @ spacing)?

A floor PLAN, a reinforcement LAYOUT drawing, a section, a beam/column schedule, or
a title block is NOT a slab schedule. Only answer true for an actual slab schedule
TABLE. Return the JSON verdict."""


# ===============================================================
# TILING
# ===============================================================

def _tile_boxes(frac=TILE_FRAC, step=STEP_FRAC):
    """Yield normalized (x1,y1,x2,y2) tiles covering the unit square with overlap."""
    boxes = []
    y = 0.0
    while y < 1.0:
        x = 0.0
        while x < 1.0:
            boxes.append((x, y, min(1.0, x + frac), min(1.0, y + frac)))
            if x + frac >= 1.0:
                break
            x += step
        if y + frac >= 1.0:
            break
        y += step
    return boxes


def _crop_tile(pil_img, box):
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    return pil_img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


# ===============================================================
# MERGE
# ===============================================================

def _overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _merge_boxes(boxes):
    """Union-find merge of overlapping/adjacent normalized boxes into clusters."""
    clusters = []
    for box in boxes:
        placed = False
        for c in clusters:
            if any(_overlaps(box, m) for m in c):
                c.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    # keep merging until stable (a box can bridge two clusters)
    merged = True
    while merged:
        merged = False
        out = []
        while clusters:
            a = clusters.pop()
            for c in out:
                if any(_overlaps(x, y) for x in a for y in c):
                    c.extend(a)
                    merged = True
                    break
            else:
                out.append(a)
        clusters = out

    regions = []
    for c in clusters:
        regions.append({
            "x1": round(min(b[0] for b in c), 4),
            "y1": round(min(b[1] for b in c), 4),
            "x2": round(max(b[2] for b in c), 4),
            "y2": round(max(b[3] for b in c), 4),
        })
    return regions


# ===============================================================
# DETECTION (overridable for testing)
# ===============================================================

def _detect_tile(tile_png_path):
    """Ask the model whether a tile contains a slab schedule table."""
    verdict = extract_structured(tile_png_path, _TILE_PROMPT, TileVerdict)
    return bool(verdict.has_slab_schedule), float(verdict.confidence)


# ===============================================================
# TOP-LEVEL
# ===============================================================

def _render_tile(page, box, out_path):
    """Render ONE normalized tile straight from the PDF at a bounded resolution."""
    W, H = page.rect.width, page.rect.height
    clip = fitz.Rect(box[0] * W, box[1] * H, box[2] * W, box[3] * H)
    long_pts = max(clip.width, clip.height) or 1.0
    # dpi chosen so the tile's long side is ~TILE_LONG_PX pixels
    dpi = int(72.0 * TILE_LONG_PX / long_pts)
    dpi = max(72, min(400, dpi))
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    pix.save(out_path)
    return out_path


def locate_slab_schedules(pdf_path, temp_folder, detect=None):
    """
    Tile the page (rendering each tile DIRECTLY from the PDF so we never build a
    giant full-page raster), detect slab-schedule tiles, merge to regions.
    `detect` can be injected for testing; defaults to the live vision detector.
    Returns a list of {region, confidence}.
    """
    detect = detect or _detect_tile
    doc = fitz.open(pdf_path)
    page = doc[0]

    tiles_dir = os.path.join(temp_folder, "_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    hits = []
    confs = []
    for i, box in enumerate(_tile_boxes()):
        tile_path = _render_tile(page, box, os.path.join(tiles_dir, f"tile_{i}.png"))
        ok, conf = detect(tile_path)
        if ok and conf >= MIN_TILE_CONF:
            hits.append(box)
            confs.append(conf)

    if not hits:
        return []

    # Do NOT merge every hit into one bounding box — on a big sheet that produces
    # a giant mostly-empty region. Instead merge only tiles that overlap a LOT
    # (same table seen in adjacent tiles), and return each cluster separately so
    # the caller can ink-guard + pattern-classify each one and drop false positives.
    clusters = _cluster_tight(hits)
    out = []
    for c in clusters:
        out.append({
            "region": {
                "x1": round(min(b[0] for b in c), 4),
                "y1": round(min(b[1] for b in c), 4),
                "x2": round(max(b[2] for b in c), 4),
                "y2": round(max(b[3] for b in c), 4),
            },
            "confidence": round(sum(confs) / len(confs), 3),
        })
    return out


def _cluster_tight(boxes, min_iou=0.4):
    """Group boxes that overlap heavily (same table in adjacent tiles)."""
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua else 0.0

    clusters = []
    for box in boxes:
        for c in clusters:
            if any(iou(box, m) >= min_iou for m in c):
                c.append(box)
                break
        else:
            clusters.append([box])
    return clusters
