"""
auto_runner.py - Slab project

Router with two extraction engines:

  ENGINE A (vector)  -> vector_extractor.py  : AutoCAD-exported / plotted PDFs.
                        Reads the schedule straight from the PDF text. No vision,
                        no hallucination. (main_1..9 not needed.)

  ENGINE B (scanned) -> sliding_window.py    : scanned/raster PDFs. Tiles the
                        high-DPI page, finds the slab-schedule region by vision,
                        crops it, then runs the normal classify -> main_1..9 path.

Routing is automatic: a page with real selectable text -> Engine A, else Engine B.
"""

import os
import json
import importlib

import fitz  # PyMuPDF

from config import INPUT_DIR, OUTPUT_DIR
import vector_extractor
import sliding_window
from table_locator import snap_region
from pdf_to_images import crop_region_to_pdf
from pattern_detector import (
    extract_layout_features, classify_slab_pattern, _ink_fraction, MIN_TABLE_INK,
)


# ===============================================================
# SHARED HELPERS
# ===============================================================

def run_pattern(pattern_number, pdf_path):
    """Run the (untouched) main_X.py extractor on a slab-only PDF crop."""
    module_name = f"main_{pattern_number}"
    print(f"🚀 Running {module_name}.py on {os.path.basename(pdf_path)}")
    try:
        module = importlib.import_module(module_name)
        module.process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to run {module_name}: {e}")


def _looks_fabricated(slabs):
    """Flag the hallucination signature: many identical rows / perfect S1..SN run."""
    n = len(slabs)
    if n < 8:
        return None

    def sig(s):
        return (s.get("thickness"), str(s.get("type", "")).strip().upper(),
                json.dumps(s.get("reinforcement", {}), sort_keys=True))

    if len({sig(s) for s in slabs}) == 1:
        return f"all {n} rows are identical"
    ids = [str(s.get("slab_id", "")).strip().upper() for s in slabs]
    if n >= 20 and all(ids[i] == f"S{i + 1}" for i in range(n)):
        return f"perfect S1..S{n} sequence ({n} rows)"
    return None


def _out_dir(base):
    d = os.path.join(OUTPUT_DIR, base)
    os.makedirs(d, exist_ok=True)
    return d


def _capped_page_png(pdf_path, out_png, max_long_px=4000):
    """
    Render the page to PNG with its long side capped at max_long_px. Used as the
    base image for OpenCV table-snapping. Avoids the multi-hundred-megapixel raster
    that a 400-DPI render of an A0 sheet would produce (PIL/cv2 choke on those).
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    long_pts = max(page.rect.width, page.rect.height) or 1.0
    dpi = max(72, min(300, int(72.0 * max_long_px / long_pts)))
    page.get_pixmap(dpi=dpi).save(out_png)
    return out_png


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ===============================================================
# ENGINE A - VECTOR
# ===============================================================

def run_vector(pdf, pdf_path):
    schedules = vector_extractor.extract_pdf(pdf_path)
    base = os.path.splitext(pdf)[0]

    if not schedules:
        print("   ⏭  No slab schedule text found on this vector sheet.")
        _write_json(os.path.join(_out_dir(base), "detection.json"),
                    {"engine": "vector", "schedules": []})
        return

    print(f"   ✅ {len(schedules)} slab schedule(s) read directly from PDF text.")
    record = {"engine": "vector", "schedules": []}
    for i, sched in enumerate(schedules, start=1):
        slabs = sched["slabs"]
        sub = f"{base}__slab_{i}"
        out_dir = _out_dir(sub)
        _write_json(os.path.join(out_dir, f"{sub}.json"), {"slabs": slabs})

        fake = _looks_fabricated(slabs)
        flag = f"  🚨 {fake}" if fake else ""
        print(f"   📄 schedule #{i} '{sched['title']}': {len(slabs)} slabs "
              f"({', '.join(s['slab_id'] for s in slabs)}){flag}")
        record["schedules"].append({
            "title": sched["title"], "region": sched["region"],
            "slab_count": len(slabs), "suspect": fake,
        })
    _write_json(os.path.join(_out_dir(base), "detection.json"), record)


# ===============================================================
# ENGINE B - SCANNED (vision sliding window)
# ===============================================================

def run_scanned(pdf, pdf_path):
    print("   🛰  Scanned sheet — locating slab schedule by sliding-window vision...")
    try:
        found = sliding_window.locate_slab_schedules(pdf_path, OUTPUT_DIR)
    except Exception as e:
        print(f"   ❌ Sliding-window localization failed: {e}")
        return

    base = os.path.splitext(pdf)[0]
    if not found:
        print("   ⏭  No slab schedule found on the scanned sheet.")
        _write_json(os.path.join(_out_dir(base), "detection.json"),
                    {"engine": "scanned", "regions": []})
        return

    print(f"   ✅ {len(found)} candidate region(s) — cropping + extracting each.")
    record = {"engine": "scanned", "targets": []}
    # size-capped page image just for OpenCV snapping (full-DPI A0 would be ~1e9 px)
    try:
        page_image = _capped_page_png(pdf_path, os.path.join(_out_dir(base), "_snapbase.png"))
    except Exception:
        page_image = None

    pg = fitz.open(pdf_path)[0]
    Wp, Hp = pg.rect.width, pg.rect.height

    for i, hit in enumerate(found, start=1):
        region = hit["region"]
        snapped = snap_region(page_image, region) if page_image else None
        crop_region = snapped or region
        # cap crop DPI so even a large (un-snapped) region on an A0 sheet stays
        # well under PIL's decompression-bomb limit while staying crisp.
        long_pts = max((crop_region["x2"] - crop_region["x1"]) * Wp,
                       (crop_region["y2"] - crop_region["y1"]) * Hp) or 1.0
        crop_dpi = max(150, min(400, int(72.0 * 3500 / long_pts)))
        crop_png, crop_pdf = crop_region_to_pdf(
            pdf_path, crop_region, f"{base}__slab_{i}", OUTPUT_DIR,
            pad=0.006 if snapped else 0.04, dpi=crop_dpi,
        )

        ink = _ink_fraction(crop_png)
        if ink is not None and ink < MIN_TABLE_INK:
            print(f"   ⏭  region #{i}: near-empty crop (ink={ink}) — skipping.")
            record["targets"].append({"region": crop_region, "ink": ink, "skipped": True})
            continue

        features = extract_layout_features(crop_png)
        cls = classify_slab_pattern(features)
        print(f"   🔎 region #{i} → Pattern {cls['pattern']} "
              f"(method={cls['method']}, conf={cls['confidence']}, ink={ink})")
        if cls["pattern"] is None:
            record["targets"].append({"region": crop_region, "pattern": None})
            continue

        run_pattern(cls["pattern"], crop_pdf)
        sub = os.path.splitext(os.path.basename(crop_pdf))[0]
        out_json = os.path.join(OUTPUT_DIR, sub, f"{sub}.json")
        if os.path.exists(out_json):
            try:
                data = json.load(open(out_json, encoding="utf-8"))
                fake = _looks_fabricated(data.get("slabs", []))
                if fake:
                    print(f"   🚨 SUSPECT OUTPUT — {fake}.")
                    open(os.path.join(OUTPUT_DIR, sub, f"{sub}.SUSPECT.txt"), "w").write(fake)
            except Exception:
                pass
        record["targets"].append({"region": crop_region, "pattern": cls["pattern"],
                                  "confidence": cls["confidence"]})

    _write_json(os.path.join(_out_dir(base), "detection.json"), record)


# ===============================================================
# MAIN
# ===============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf)
        is_vector = vector_extractor.pdf_is_vector(pdf_path)
        engine = "VECTOR (pdf text)" if is_vector else "SCANNED (vision)"
        print(f"\n📄 {pdf}\n   Engine: {engine}")

        try:
            if is_vector:
                run_vector(pdf, pdf_path)
            else:
                run_scanned(pdf, pdf_path)
        except Exception as e:
            print(f"   ❌ Failed for {pdf}: {e}")


if __name__ == "__main__":
    main()
