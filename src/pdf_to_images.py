import fitz  # PyMuPDF
import os

def convert_pdf_to_images(pdf_path, output_folder):

    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):

        pix = page.get_pixmap(dpi=300)

        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        pix.save(img_path)

        image_paths.append(img_path)

    return image_paths
