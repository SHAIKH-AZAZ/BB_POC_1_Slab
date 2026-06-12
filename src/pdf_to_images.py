import fitz  # PyMuPDF
import os


def convert_pdf_to_images(pdf_path, output_folder):
    """Render every page of a PDF to a PNG at 300 dpi. Returns list of paths."""
    doc = fitz.open(pdf_path)
    image_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    return image_paths


def _clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def render_pdf_region_to_png(pdf_path, region, out_png,
                             page_index=0, pad=0.03, dpi=400):
    """
    Crop a NORMALIZED (0.0-1.0) region of a PDF page and render it to PNG.

    `region` is a dict {x1, y1, x2, y2} in page-relative fractions. `pad` adds a
    margin on every side (so an imprecise model bounding box doesn't clip rows).
    Returns out_png.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    r = page.rect

    x1 = _clamp01(region.get("x1", 0.0) - pad)
    y1 = _clamp01(region.get("y1", 0.0) - pad)
    x2 = _clamp01(region.get("x2", 1.0) + pad)
    y2 = _clamp01(region.get("y2", 1.0) + pad)
    if x2 <= x1:
        x1, x2 = 0.0, 1.0
    if y2 <= y1:
        y1, y2 = 0.0, 1.0

    clip = fitz.Rect(r.width * x1, r.height * y1, r.width * x2, r.height * y2)
    pix = page.get_pixmap(dpi=dpi, clip=clip)
    pix.save(out_png)
    doc.close()
    return out_png


def png_to_pdf(png_path, out_pdf):
    """
    Wrap a single PNG into a one-page PDF whose page exactly fits the image.
    Lets the existing main_1..9 extractors (which take a PDF) run on an isolated
    cropped table without any changes to them.
    """
    src = fitz.open(png_path)
    pdf_bytes = src.convert_to_pdf()
    src.close()
    with open(out_pdf, "wb") as fh:
        fh.write(pdf_bytes)
    return out_pdf


def crop_region_to_pdf(pdf_path, region, out_basename, temp_folder,
                       page_index=0, pad=0.03, dpi=400):
    """
    Convenience: crop a normalized region of a PDF and produce BOTH a PNG (for
    vision feature/pattern detection) and a one-page PDF (for extraction).
    Returns (png_path, pdf_path).
    """
    os.makedirs(temp_folder, exist_ok=True)
    png_path = os.path.join(temp_folder, f"{out_basename}.png")
    pdf_out = os.path.join(temp_folder, f"{out_basename}.pdf")
    render_pdf_region_to_png(pdf_path, region, png_path,
                             page_index=page_index, pad=pad, dpi=dpi)
    png_to_pdf(png_path, pdf_out)
    return png_path, pdf_out
