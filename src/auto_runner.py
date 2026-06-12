import os
import json
import importlib

from config import INPUT_DIR, OUTPUT_DIR
from pattern_detector import detect_document


def run_pattern(pattern_number, pdf_path):
    """
    Dynamically import the correct main_X.py extractor (left untouched) and run it
    on the given PDF (which, for multi-schedule sheets, is the cropped slab-only PDF).
    """
    module_name = f"main_{pattern_number}"
    print(f"🚀 Running {module_name}.py on {os.path.basename(pdf_path)}")

    try:
        module = importlib.import_module(module_name)
        module.process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to run {module_name}: {e}")


def _write_detection_record(pdf, detection):
    """Persist the full detection result for auditing (what was found / skipped)."""
    file_name = os.path.splitext(os.path.basename(pdf))[0]
    folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, "detection.json")
    # features can be bulky; keep the record readable but complete.
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(detection, fh, indent=2)
    print(f"📝 Detection record saved to {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf)
        print(f"\n📄 Detecting schedules in {pdf}...")

        try:
            detection = detect_document(pdf_path, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Detection failed for {pdf}: {e}")
            continue

        found = [f"{s['document_type']}" for s in detection["schedules"]]
        print(f"   Schedules on sheet: {found or 'none'}")

        slab_targets = detection["slab_targets"]
        if not slab_targets:
            print(f"⏭  No slab schedule on {pdf} — nothing to extract.")
            _write_detection_record(pdf, detection)
            continue

        print(f"   ✅ {len(slab_targets)} slab schedule(s) found — extracting each.")
        for tgt in slab_targets:
            label = f"slab #{tgt['index']} ('{tgt['title'] or 'untitled'}')"
            if tgt["pattern"] is None:
                print(f"   ⚠ {label}: no pattern (1-9) matched — skipping. See detection.json.")
                continue
            print(
                f"   🔎 {label} → Pattern {tgt['pattern']} "
                f"(method={tgt['method']}, confidence={tgt['pattern_confidence']})"
            )
            run_pattern(tgt["pattern"], tgt["pdf_path"])

        _write_detection_record(pdf, detection)


if __name__ == "__main__":
    main()
