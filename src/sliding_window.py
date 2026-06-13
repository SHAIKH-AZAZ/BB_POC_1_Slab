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

import table_locator
from image_enhance import enhance_for_vision
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


def _capped_page_png(pdf_path, temp_folder, max_long=3000):
    """Render the full page with its long side capped (for OpenCV table detection)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    long_pts = max(page.rect.width, page.rect.height) or 1.0
    dpi = max(72, min(300, int(72.0 * max_long / long_pts)))
    out = os.path.join(temp_folder, "_page_for_tables.png")
    page.get_pixmap(dpi=dpi).save(out)
    return out


def _render_region_png(page, region, out_path, max_long=2000, pad=0.012):
    """Render a normalized region from the PDF at a bounded resolution, enhanced."""
    W, H = page.rect.width, page.rect.height
    x1 = max(0.0, region["x1"] - pad); y1 = max(0.0, region["y1"] - pad)
    x2 = min(1.0, region["x2"] + pad); y2 = min(1.0, region["y2"] + pad)
    clip = fitz.Rect(x1 * W, y1 * H, x2 * W, y2 * H)
    long_pts = max(clip.width, clip.height) or 1.0
    dpi = max(72, min(300, int(72.0 * max_long / long_pts)))
    page.get_pixmap(dpi=dpi, clip=clip).save(out_path)
    enhance_for_vision(out_path)   # darken faint gray so the model can read it
    return out_path


def locate_slab_schedules(pdf_path, temp_folder, detect=None):
    """
    Locate slab-schedule region(s) on a sheet with no usable text.

    PRIMARY (deterministic): detect ruled table rectangles with OpenCV (now
    light-gray aware), then ask the model a cheap yes/no per candidate table.
    This is precise and cheap on the common case (ruled CAD tables).

    FALLBACK (vision only): if no ruled tables are found (border-less or noisy
    scan), tile the page and detect per tile.

    `detect` (tile -> (bool, conf)) is injectable for testing.
    Returns a list of {region, confidence}.
    """
    detect = detect or _detect_tile
    os.makedirs(temp_folder, exist_ok=True)

    # ---- PRIMARY: OpenCV ruled-table candidates ----
    try:
        page_png = _capped_page_png(pdf_path, temp_folder)
        candidates = table_locator.detect_table_boxes(page_png)
    except Exception:
        candidates = []

    if candidates:
        page = fitz.open(pdf_path)[0]
        cand_dir = os.path.join(temp_folder, "_cand")
        os.makedirs(cand_dir, exist_ok=True)
        hits = []
        for i, region in enumerate(candidates):
            png = _render_region_png(page, region, os.path.join(cand_dir, f"cand_{i}.png"))
            try:
                ok, conf = detect(png)
            except Exception:
                ok, conf = False, 0.0
            if ok and conf >= MIN_TILE_CONF:
                hits.append({"region": region, "confidence": round(float(conf), 3)})
        if hits:
            return hits

    # ---- FALLBACK: sliding-window tiling ----
    return _tiling_locate(pdf_path, temp_folder, detect)


def _tiling_locate(pdf_path, temp_folder, detect):
    """Render overlapping tiles directly from the PDF and detect per tile."""
    page = fitz.open(pdf_path)[0]
    tiles_dir = os.path.join(temp_folder, "_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    hits, confs = [], []
    for i, box in enumerate(_tile_boxes()):
        tile_path = _render_tile(page, box, os.path.join(tiles_dir, f"tile_{i}.png"))
        ok, conf = detect(tile_path)
        if ok and conf >= MIN_TILE_CONF:
            hits.append(box)
            confs.append(conf)

    if not hits:
        return []

    # Merge only tiles that overlap heavily (same table seen twice); return each
    # cluster separately so the caller ink-guards + pattern-classifies each and
    # drops false positives, instead of collapsing into one giant empty region.
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
