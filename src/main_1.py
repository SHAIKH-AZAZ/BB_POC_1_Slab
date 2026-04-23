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
    with open(os.path.join(os.path.dirname(__file__), "prompt_1.txt"), "r") as f:
        return f.read()


# ==============================
# CLEAN SLAB DATA
# ==============================

def clean_slab(slab):

    # Deduplicate dia
    slab["reinforcement"]["dia"] = sorted(
        list(set(slab["reinforcement"].get("dia", [])))
    )

    # Deduplicate spacing
    slab["reinforcement"]["spacing"] = sorted(
        list(set(slab["reinforcement"].get("spacing", [])))
    )

    return slab


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

    cleaned_slabs = []

    for slab in all_slabs:
        cleaned_slabs.append(clean_slab(slab))

    final_output = {"slabs": cleaned_slabs}

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
        print("⚠ No PDF files found in input folder.")
        return

    for pdf in pdf_files:
        process_pdf(os.path.join(INPUT_DIR, pdf))


if __name__ == "__main__":
    main()
