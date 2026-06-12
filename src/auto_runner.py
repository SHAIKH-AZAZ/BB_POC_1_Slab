import os
import json
import importlib

from config import INPUT_DIR, OUTPUT_DIR
from pattern_detector import detect_document, SLAB_FAMILY


def run_pattern(pattern_number, pdf_path):
    """
    Dynamically import the correct main_X.py extractor (left untouched).
    """
    module_name = f"main_{pattern_number}"
    print(f"🚀 Running {module_name}.py")

    try:
        module = importlib.import_module(module_name)
        module.process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to run {module_name}: {e}")


def _write_skip_record(pdf, detection):
    """
    Persist a small JSON for drawings we did NOT extract (non-slab families, or
    slab schedules whose pattern could not be matched). Keeps the run auditable.
    """
    file_name = os.path.splitext(os.path.basename(pdf))[0]
    folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, "detection.json")
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
        print(f"\n📄 Detecting document type for {pdf}...")

        try:
            detection = detect_document(pdf_path, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Detection failed for {pdf}: {e}")
            continue

        doc_type = detection["document_type"]

        # Stage 1 gate: not a slab schedule -> do NOT force a pattern.
        if doc_type != SLAB_FAMILY:
            print(
                f"⏭  Skipping {pdf}: detected {doc_type} "
                f"(confidence {detection['family_confidence']}), not a slab schedule."
            )
            _write_skip_record(pdf, detection)
            continue

        pattern_number = detection["pattern"]
        if pattern_number is None:
            print(
                f"⚠ {pdf} is a slab schedule but no pattern (1-9) matched its layout. "
                "Skipping extraction; review detection.json."
            )
            _write_skip_record(pdf, detection)
            continue

        print(
            f"🔎 Slab schedule → Pattern {pattern_number} "
            f"(method={detection['method']}, confidence={detection['pattern_confidence']})"
        )
        run_pattern(pattern_number, pdf_path)


if __name__ == "__main__":
    main()
