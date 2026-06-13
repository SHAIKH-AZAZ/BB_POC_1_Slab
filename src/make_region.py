"""
make_region.py - interactive slab-table region picker.

Draw a box around each slab schedule on a sheet and it saves the normalized
coordinates to input/regions/<pdfname>.json, which auto_runner then uses as a
GUARANTEED region hint (no auto-detection needed) for that sheet.

The preview is contrast-enhanced so even very faint (light-gray) tables are
visible enough to click.

Usage (run on your own machine, where a screen is available):
    python src/make_region.py                         # pick a PDF from input/
    python src/make_region.py "input/my sheet.pdf"    # open a specific PDF

Controls:
    - drag a rectangle around a slab schedule (repeat for multiple tables)
    - U / Undo button : remove the last box
    - S / Save button : write the regions JSON
    - Q / Esc         : quit
"""

import json
import os
import sys

import fitz  # PyMuPDF
from PIL import Image

# project layout (this file lives in <root>/src)
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
INPUT_DIR = os.path.join(_ROOT, "input")
REGIONS_DIR = os.path.join(INPUT_DIR, "regions")

# optional contrast enhancement so faint tables are visible
try:
    sys.path.insert(0, _SRC)
    from image_enhance import enhance_for_vision
except Exception:
    enhance_for_vision = None


# ===============================================================
# CORE (headless-testable) HELPERS
# ===============================================================

def resolve_pdf(arg=None):
    """Return a PDF path from an arg, or by listing input/ and prompting."""
    if arg:
        return arg if os.path.isabs(arg) else os.path.join(_ROOT, arg) \
            if not os.path.exists(arg) else arg
    pdfs = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise SystemExit(f"No PDFs in {INPUT_DIR}")
    if len(pdfs) == 1:
        return os.path.join(INPUT_DIR, pdfs[0])
    print("Select a PDF:")
    for i, f in enumerate(pdfs):
        print(f"  [{i}] {f}")
    idx = int(input("number> ").strip())
    return os.path.join(INPUT_DIR, pdfs[idx])


def render_preview(pdf_path, max_w=1300, max_h=820, enhance=True):
    """
    Render page 0 to a PIL image scaled to fit (max_w x max_h). Returns
    (pil_image, disp_w, disp_h). The display size == image size, so screen
    pixel coords map to fractions by dividing by disp_w/disp_h.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    Wp, Hp = page.rect.width, page.rect.height
    # dpi so BOTH dimensions fit the window. No high floor: a huge A0 sheet needs a
    # LOW dpi to fit, so only a tiny floor (avoid 0) and a cap for small sheets.
    dpi_w = 72.0 * max_w / Wp
    dpi_h = 72.0 * max_h / Hp
    dpi = max(6, min(200, int(min(dpi_w, dpi_h))))
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(),
                       f"slab_preview_{os.getpid()}.png")
    page.get_pixmap(dpi=dpi).save(tmp)
    if enhance and enhance_for_vision:
        enhance_for_vision(tmp)
    img = Image.open(tmp).convert("RGB")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return img, img.width, img.height


def normalize_box(px_box, disp_w, disp_h):
    """Pixel box (x1,y1,x2,y2) on the displayed image -> normalized dict (0-1)."""
    x1, y1, x2, y2 = px_box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    clamp = lambda v, m: max(0.0, min(1.0, v / m)) if m else 0.0
    return {
        "x1": round(clamp(x1, disp_w), 4), "y1": round(clamp(y1, disp_h), 4),
        "x2": round(clamp(x2, disp_w), 4), "y2": round(clamp(y2, disp_h), 4),
    }


def save_regions(pdf_path, regions):
    """Write input/regions/<base>.json (single dict if one, else a list)."""
    os.makedirs(REGIONS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out = os.path.join(REGIONS_DIR, f"{base}.json")
    payload = regions[0] if len(regions) == 1 else regions
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return out


# ===============================================================
# GUI
# ===============================================================

def run_gui(pdf_path):
    import tkinter as tk

    img, disp_w, disp_h = render_preview(pdf_path)
    try:
        from PIL import ImageTk
    except Exception as e:
        raise SystemExit(f"Pillow ImageTk required for the GUI: {e}")

    root = tk.Tk()
    root.title(f"Pick slab regions — {os.path.basename(pdf_path)}")
    regions = []        # normalized dicts
    rect_ids = []       # canvas rectangle ids (persistent)

    info = tk.Label(root, text="Drag a box around each slab schedule.  "
                               "[S]ave   [U]ndo   [Q]uit", anchor="w")
    info.pack(fill="x")

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="cross")
    canvas.pack()
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

    state = {"x0": 0, "y0": 0, "live": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        state["live"] = canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                outline="red", width=2)

    def on_drag(e):
        if state["live"] is not None:
            canvas.coords(state["live"], state["x0"], state["y0"], e.x, e.y)

    def on_release(e):
        if state["live"] is None:
            return
        box = (state["x0"], state["y0"], e.x, e.y)
        if abs(box[2] - box[0]) < 6 or abs(box[3] - box[1]) < 6:
            canvas.delete(state["live"])           # ignore tiny accidental clicks
        else:
            regions.append(normalize_box(box, disp_w, disp_h))
            rect_ids.append(state["live"])
            n = len(regions)
            canvas.create_text(box[0] + 4, box[1] + 8, anchor="w",
                               text=f"#{n}", fill="red")
            info.config(text=f"{n} region(s).  [S]ave   [U]ndo   [Q]uit")
        state["live"] = None

    def undo(_e=None):
        if regions:
            regions.pop()
            canvas.delete(rect_ids.pop())
            info.config(text=f"{len(regions)} region(s).  [S]ave   [U]ndo   [Q]uit")

    def save(_e=None):
        if not regions:
            info.config(text="Draw at least one box before saving.")
            return
        out = save_regions(pdf_path, regions)
        info.config(text=f"Saved {len(regions)} region(s) -> {out}")
        print("Saved:", out)

    def quit_(_e=None):
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("s", save); root.bind("S", save)
    root.bind("u", undo); root.bind("U", undo)
    root.bind("q", quit_); root.bind("Q", quit_); root.bind("<Escape>", quit_)

    btns = tk.Frame(root); btns.pack(fill="x")
    tk.Button(btns, text="Save", command=save).pack(side="left")
    tk.Button(btns, text="Undo", command=undo).pack(side="left")
    tk.Button(btns, text="Quit", command=quit_).pack(side="right")

    root.mainloop()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    pdf_path = resolve_pdf(arg)
    if not os.path.exists(pdf_path):
        raise SystemExit(f"Not found: {pdf_path}")
    print(f"Opening {pdf_path} …  draw boxes, press S to save.")
    run_gui(pdf_path)


if __name__ == "__main__":
    main()
