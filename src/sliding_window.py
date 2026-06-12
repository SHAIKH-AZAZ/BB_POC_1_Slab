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

from pydantic import BaseModel, Field

from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_structured


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

def locate_slab_schedules(pdf_path, temp_folder, dpi=300, detect=None):
    """
    Tile the (rendered) page, detect slab-schedule tiles, merge to regions.
    `detect` can be injected for testing; defaults to the live vision detector.
    Returns a list of {region, confidence}.
    """
    from PIL import Image

    detect = detect or _detect_tile
    page_img = convert_pdf_to_images(pdf_path, temp_folder)[0]
    img = Image.open(page_img).convert("RGB")

    tiles_dir = os.path.join(temp_folder, "_tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    hits = []
    confs = []
    for i, box in enumerate(_tile_boxes()):
        tile = _crop_tile(img, box)
        tile_path = os.path.join(tiles_dir, f"tile_{i}.png")
        tile.save(tile_path)
        ok, conf = detect(tile_path)
        if ok and conf >= MIN_TILE_CONF:
            hits.append(box)
            confs.append(conf)

    if not hits:
        return []

    regions = _merge_boxes(hits)
    avg = round(sum(confs) / len(confs), 3)
    return [{"region": r, "confidence": avg} for r in regions]
