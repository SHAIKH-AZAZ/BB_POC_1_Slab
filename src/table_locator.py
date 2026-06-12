"""
table_locator.py - Slab project

Deterministic table localization with OpenCV.

The vision model is good at *identifying* and *ordering* the schedules on a sheet
but unreliable at precise pixel bounding boxes (it tends to be shifted/clipped).
Engineering schedules, however, are drawn as RULED tables with strong borders, so
their exact rectangles can be recovered from the image with classic line
detection. We then snap the model's approximate region to the real table rectangle
so a crop never clips rows.

Falls back gracefully (returns None) when no ruled table is found - e.g. a scanned
or border-less table - so the caller can use the padded model region instead.
"""

try:
    import cv2
    import numpy as np
    _CV_OK = True
except Exception:  # cv2 not installed -> snapping disabled, caller falls back
    _CV_OK = False


def detect_table_boxes(image_path):
    """
    Find ruled-table rectangles in a page image.
    Returns a list of normalized boxes [{x1,y1,x2,y2}, ...] (0.0-1.0), leaf tables
    only (the outer sheet frame and big enclosing rectangles are removed).
    Returns [] if OpenCV is unavailable or the image can't be read.
    """
    if not _CV_OK:
        return []
    img = cv2.imread(image_path)
    if img is None:
        return []

    # Normalize to a fixed working width so the morphology kernels behave the same
    # regardless of source DPI (at very high res the kernels erode thin table rules
    # and small tables disappear). Output is normalized, so the resize is lossless
    # for our purposes.
    TARGET_W = 2200
    h0, w0 = img.shape[:2]
    if w0 > TARGET_W:
        img = cv2.resize(img, (TARGET_W, int(h0 * TARGET_W / w0)),
                         interpolation=cv2.INTER_AREA)

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, -2
    )

    # isolate long horizontal and vertical rules
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 40)))
    horiz = cv2.dilate(cv2.erode(th, hk), hk)
    vert = cv2.dilate(cv2.erode(th, vk), vk)
    grid = cv2.add(horiz, vert)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # plausible table: reasonably wide & tall, not the full-sheet frame
        if w > W * 0.12 and h > H * 0.03 and w * h > W * H * 0.008:
            if w > W * 0.95 and h > H * 0.95:
                continue
            raw.append((x, y, x + w, y + h))

    # drop "enclosing" rectangles that contain the centers of >=2 other boxes
    def contains_center(outer, inner):
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

    leaves = []
    for a in raw:
        enclosed = sum(
            1 for b in raw if b is not a and contains_center(a, b)
        )
        if enclosed < 2:
            leaves.append(a)

    return [
        {"x1": x1 / W, "y1": y1 / H, "x2": x2 / W, "y2": y2 / H}
        for (x1, y1, x2, y2) in leaves
    ]


def _x_overlap_frac(a, b):
    inter = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    if inter <= 0:
        return 0.0
    return inter / min(a["x2"] - a["x1"], b["x2"] - b["x1"])


def snap_region(image_path, region, min_x_overlap=0.3):
    """
    Snap an approximate (model) region to the real ruled-table rectangle.

    Strategy tolerant to the model's vertical offset: among detected tables that
    horizontally overlap the model region, pick the one whose vertical CENTER is
    nearest the model region's center (relative ordering is preserved even when
    the absolute box is shifted). Returns a normalized box, or None if no ruled
    table matches (caller should fall back to the padded model region).
    """
    boxes = detect_table_boxes(image_path)
    if not boxes:
        return None

    model_cy = (region["y1"] + region["y2"]) / 2
    candidates = [b for b in boxes if _x_overlap_frac(region, b) >= min_x_overlap]
    if not candidates:
        return None

    best = min(candidates, key=lambda b: abs(((b["y1"] + b["y2"]) / 2) - model_cy))
    return {k: round(v, 4) for k, v in best.items()}
