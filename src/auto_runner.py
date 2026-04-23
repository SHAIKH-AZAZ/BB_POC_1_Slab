import os
import importlib

from config import INPUT_DIR, OUTPUT_DIR
from pattern_detector import detect_pattern


def run_pattern(pattern_number, pdf_path):
    """
    Dynamically import correct main_X.py
    """

    module_name = f"main_{pattern_number}"

    print(f"🚀 Running {module_name}.py")

    try:
        module = importlib.import_module(module_name)
        module.process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to run {module_name}: {e}")


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:

        pdf_path = os.path.join(INPUT_DIR, pdf)

        print(f"\n📄 Detecting pattern for {pdf}...")

        try:
            pattern_number = detect_pattern(pdf_path, OUTPUT_DIR)
            print(f"🔎 Detected Pattern: {pattern_number}")

            run_pattern(pattern_number, pdf_path)

        except Exception as e:
            print(f"❌ Failed for {pdf}: {e}")


if __name__ == "__main__":
    main()