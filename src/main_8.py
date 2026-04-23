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
    with open(os.path.join(os.path.dirname(__file__), "prompt_8.txt"), "r") as f:
        return f.read()


# ==============================
# CLEAN REINFORCEMENT
# ==============================

def clean_reinforcement(reinf):

    dia = set()
    spacing = set()

    for d in reinf.get("dia", []):
        if d:
            dia.add(d.strip().upper())

    for s in reinf.get("spacing", []):
        if s:
            s = s.strip().upper()

            # normalize spacing
            if "C/C" not in s:
                s_clean = s.replace(" ", "")
                if s_clean.isdigit():
                    s = f"{s_clean} C/C"

            spacing.add(s)

    return {
        "dia": sorted(list(dia)),
        "spacing": sorted(list(spacing))
    }


# ==============================
# PROCESS PDF
# ==============================

def process_pdf(pdf_path):

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_output_folder = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    print(f"\n📄 Converting {file_name}.pdf to images...")
    image_paths = convert_pdf_to_images(pdf_path, file_output_folder)

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

    # ==============================
    # DEDUPLICATION
    # ==============================

    unique_slabs = {}

    for slab in all_slabs:

        slab_id = slab.get("slab_id")
        if not slab_id:
            continue

        if slab_id not in unique_slabs:
            unique_slabs[slab_id] = slab
        else:
            existing = unique_slabs[slab_id]

            existing["reinforcement"]["dia"] += slab["reinforcement"].get("dia", [])
            existing["reinforcement"]["spacing"] += slab["reinforcement"].get("spacing", [])

            unique_slabs[slab_id] = existing

    # ==============================
    # FINAL CLEAN
    # ==============================

    final_slabs = []

    for slab in unique_slabs.values():

        slab["reinforcement"] = clean_reinforcement(
            slab.get("reinforcement", {})
        )

        final_slabs.append(slab)

    final_output = {"slabs": final_slabs}

    output_file = os.path.join(file_output_folder, f"{file_name}.json")

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