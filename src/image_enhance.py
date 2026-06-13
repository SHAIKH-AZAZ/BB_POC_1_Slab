"""
image_enhance.py - Slab project

Contrast tools for FAINT CAD plots.

Many AutoCAD PDFs draw tables and text in light gray (~180-230), not black. A
"darker than 160" rule treats those as blank, which broke both OpenCV table
detection and the blank-crop guard. The helpers here threshold/enhance RELATIVE
to the page's own background, so they work on light-gray vector plots, near-black
linework, and slightly off-white scans alike.

    ink_threshold(gray)       -> luminance cutoff just below paper-white
    ink_mask(gray)            -> uint8 mask (255 = ink) for OpenCV
    ink_fraction(path|img)    -> fraction of non-background pixels (blank guard)
    enhance_for_vision(path)  -> darken faint content, keep grayscale (vision read)

Pure NumPy/Pillow (no OpenCV dependency) so it always imports.
"""

import numpy as np
from PIL import Image, ImageOps


# ===============================================================
# LOADING
# ===============================================================

def _to_gray(image, max_side=None):
    """Accept a path, PIL image, or ndarray; return a uint8 grayscale ndarray."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = np.asarray(Image.fromarray(arr).convert("L"))
        return arr
    im = Image.open(image) if isinstance(image, str) else image
    im = im.convert("L")
    if max_side:
        im.thumbnail((max_side, max_side))
    return np.asarray(im)


# ===============================================================
# BACKGROUND-RELATIVE THRESHOLDING
# ===============================================================

def background_level(gray):
    """Estimate the paper/background luminance (the bright peak ~ white)."""
    return int(np.percentile(gray, 99))


def ink_threshold(gray, delta=30, lo=120, hi=245):
    """
    Luminance cutoff: pixels darker than this are 'ink'. Set just below the page
    background so light-gray content counts, while clean white (255) does not.
    `lo`/`hi` clamp it to a sane range so a freak histogram can't disable it.
    """
    bg = background_level(gray)
    return int(np.clip(bg - delta, lo, hi))


def ink_mask(image, delta=30):
    """
    Binary ink mask as uint8 (255 = ink, 0 = background), background-relative.
    Returns (mask, threshold_used). `image` may be a path, PIL image, or ndarray.
    """
    gray = _to_gray(image)
    thr = ink_threshold(gray, delta=delta)
    mask = np.where(gray < thr, np.uint8(255), np.uint8(0))
    return mask, thr


def ink_fraction(image, delta=15):
    """
    Fraction (0-1) of non-background pixels. Robust blank-crop signal that works
    on light-gray tables (unlike a fixed <160 'dark' count). A smaller `delta`
    here makes it sensitive to even faint content while clean white stays ~0.
    """
    try:
        gray = _to_gray(image, max_side=1000)
    except Exception:
        return None
    thr = ink_threshold(gray, delta=delta)
    total = gray.size
    if not total:
        return None
    return round(float(np.count_nonzero(gray < thr)) / total, 4)


# ===============================================================
# ENHANCEMENT FOR THE VISION MODEL
# ===============================================================

def enhance_for_vision(in_path, out_path=None, gamma=1.6, cutoff=0.5):
    """
    Darken faint gray linework/text while keeping smooth grayscale (the vision
    model reads natural grayscale better than harsh 1-bit, which can break thin
    reinforcement text like 'T8@150C/C'). In-place by default.

      autocontrast(cutoff) stretches the histogram so the faintest real gray maps
      toward black; gamma>1 then deepens the mid/light tones.

    Returns the output path. Never raises - on any failure the original is kept.
    """
    out_path = out_path or in_path
    try:
        im = Image.open(in_path).convert("L")
        im = ImageOps.autocontrast(im, cutoff=cutoff)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.clip(arr ** gamma, 0.0, 1.0)
        Image.fromarray((arr * 255).astype(np.uint8)).save(out_path)
    except Exception:
        if out_path != in_path:
            try:
                Image.open(in_path).save(out_path)
            except Exception:
                pass
    return out_path
