import os
import json
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR
from pdf_to_images import convert_pdf_to_images
from text_recovery import recover_and_merge
from vision_extractor import extract_from_image, extract_with_tools


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
    reinf = slab.setdefault("reinforcement", {"dia": [], "spacing": []})

    # Deduplicate + sort dia
    reinf["dia"] = sorted(set(reinf.get("dia") or []))

    # Deduplicate + sort spacing
    reinf["spacing"] = sorted(set(reinf.get("spacing") or []))

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

        result = extract_with_tools(img_path, prompt)

        try:
            parsed = json.loads(result)
            if "slabs" in parsed:
                all_slabs.extend(parsed["slabs"])
        except:
            print("⚠ JSON parse failed")

    # Pattern-1 historically returned slab_type + along/across span arrays.
    # Normalize to the canonical schema before recovery + cleanup.
    normalized = []
    for slab in all_slabs:
        slab_id = slab.get("slab_id") or slab.get("slab_type") or ""
        reinf = slab.get("reinforcement") or {}
        normalized.append({
            "slab_id": slab_id,
            "thickness": slab.get("thickness"),
            "type": slab.get("type") or "",
            "mix": slab.get("mix") or "",
            "reinforcement": {
                "dia": list(reinf.get("dia") or []),
                "spacing": list(reinf.get("spacing") or []),
            },
            "remarks": slab.get("remarks") or "",
        })

    # Backstop: re-parse the PDF's text layer to recover any spacing/dia
    # the model may have dropped.
    recover_and_merge(pdf_path, normalized)

    cleaned_slabs = [clean_slab(s) for s in normalized]

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
