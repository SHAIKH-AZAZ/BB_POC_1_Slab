import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


# ==============================
# LOAD PROMPT
# ==============================

def load_prompt():
    with open(os.path.join(os.path.dirname(__file__), "prompt_5.txt"), "r") as f:
        return f.read()


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(reinf):

    dia_set = set()
    spacing_set = set()

    for d in reinf.get("dia", []):
        if d:
            dia_set.add(d.strip())

    for s in reinf.get("spacing", []):
        if s:
            s = s.upper().replace("@", "").replace("C/C", "").replace("C", "")
            s = s.strip()
            if s.isdigit():
                spacing_set.add(f"{s} C/C")

    return {
        "dia": sorted(list(dia_set)),
        "spacing": sorted(list(spacing_set))
    }


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, output_folder)

    prompt = load_prompt()
    all_slabs = []

    for img_path in tqdm(image_paths):

        result = extract_from_image(img_path, prompt)

        try:
            parsed = json.loads(result)
            if "slabs" in parsed:
                all_slabs.extend(parsed["slabs"])
        except:
            print("⚠ JSON parse failed")

    # Clean and deduplicate
    unique = {}

    for slab in all_slabs:
        slab_id = slab.get("slab_id")
        if not slab_id:
            continue

        slab["reinforcement"] = clean_reinforcement(
            slab.get("reinforcement", {})
        )

        unique[slab_id] = slab

    final_output = {"slabs": list(unique.values())}

    output_file = os.path.join(output_folder, f"{file_name}.json")

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"✅ Output saved to {output_file}")


# ==============================
# MAIN
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠ No PDF files found.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
