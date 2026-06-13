# Slab Schedule Extraction — Architecture (AGENTS.md)

Extracts **RCC slab reinforcement schedules** from structural drawing PDFs into
JSON. Input PDFs vary wildly: clean vector exports, scanned rasters, AutoCAD
plots with text converted to outline curves, ultra-faint gray line-work, multiple
schedules per sheet, and even *transposed* tables. The system routes each sheet to
the cheapest engine that can read it and falls back when needed.

---

## 1. Big picture

```
                 ┌─────────────────────────── auto_runner.py (router) ───────────────────────────┐
                 │                                                                                │
  input/*.pdf ──▶│  region hint?  ──yes──▶  ENGINE: MANUAL  (run_manual)                          │
                 │      │no                                                                        │
                 │      ▼                                                                          │
                 │  has real text?  ──yes──▶  ENGINE A: VECTOR  (run_vector → vector_extractor)    │
                 │      │no / found nothing                          │ found slabs → write JSON    │
                 │      ▼                            └── no slabs ────┘ (fall through)              │
                 │  ENGINE B: SCANNED / VISION  (run_scanned → sliding_window + main_1..9)         │
                 └────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                            output/<pdf>__slab_<n>/<...>.json   (+ detection.json)
```

Routing priority, per PDF:

1. **Manual region hint** (`input/regions/<pdf>.json`) — if present, wins. Guaranteed.
2. **Engine A (vector)** — if the page has real selectable text. Reads the schedule
   straight from PDF text. No vision, no API, cannot hallucinate.
3. **Engine B (vision)** — scanned / outline-curve / faint sheets, *or* when Engine A
   finds no slab schedule (mixed PDFs).

The deterministic engines are preferred because they are free, instant, and exact.
Vision is the fallback for images.

---

## 2. The two engines

### Engine A — vector (`vector_extractor.py`)

For PDFs whose text is real characters (`page.get_text()` returns words with
coordinates).

1. **Locate** the schedule by **fuzzy title match**: any heading line containing a
   SLAB word + a SCHEDULE/DETAILS/REINFORCEMENT word, in any order
   (`SCHEDULE OF RCC SLABS`, `SLAB REINFORCEMENT SCHEDULE`, `R.C.C. SLAB DETAILS` …).
   Beam/column/footing headings are rejected.
2. **Orientation detection**: a row carrying ≥2 slab marks (`S1 S2 S3 S4`) means the
   slabs are **column headers** → transposed table.
   - **Normal** layout → one slab per row; parse each row.
   - **Transposed** layout (`_parse_transposed`) → assign every value word to its
     nearest slab **column** by x, read down each column, parse each column as one slab.
3. **Parse fields** per slab with regexes: `slab_id`, `thickness`, `mix` (M-grade),
   `type` (ONE/TWO WAY), `reinforcement` (`dia`, `spacing` from `nT @ s c/c` etc.).

Output goes straight to JSON — `main_1..9` are not used for vector PDFs.

### Engine B — vision (`sliding_window.py` + `pattern_detector.py` + `main_1..9`)

For scanned rasters and outline-curve / faint sheets (no usable text).

1. **Localize** the slab table:
   - **Primary (deterministic):** `table_locator.detect_table_boxes` finds ruled
     table rectangles with OpenCV (background-relative threshold catches faint gray
     lines). Every ruled table is a candidate.
   - **Fallback:** sliding-window tiling — render overlapping tiles, ask the model
     per tile "is a slab schedule here?", merge hits. Used only when no ruled tables
     are found.
2. **Crop** each candidate region into a contrast-enhanced PNG + a one-page mini-PDF.
3. **Classify** (`pattern_detector`): Stage 2 reads layout features from the image
   (+ OCR text hint), Stage 3 matches them deterministically to one of 9 patterns.
   Non-slab tables (beam/column/plan) → `pattern None` → skipped.
4. **Extract** with the matching untouched `main_<pattern>.py` on the mini-PDF.

The classifier (which reads the actual column headers) is the discriminator —
there is **no** unreliable vision yes/no gate deciding which candidate to keep.

---

## 3. Modules

| File | Role |
|------|------|
| `auto_runner.py` | Entry point + router. `run_vector` / `run_scanned` / `run_manual`. Owns the engine fallback logic and the output/guard wiring. |
| `vector_extractor.py` | **Engine A.** Fuzzy title match, normal + transposed parsing, field regexes. |
| `sliding_window.py` | **Engine B localization.** OpenCV-tables-first, tiling fallback. |
| `table_locator.py` | OpenCV ruled-table detection (background-relative, light-gray aware) + `snap_region`. |
| `pattern_detector.py` | Stage 1 (multi-schedule family detect, legacy), **Stage 2** layout features (Pydantic structured output + OCR hint), **Stage 3** deterministic pattern matcher, ink guard. |
| `image_enhance.py` | Contrast tools: background-relative `ink_mask` / `ink_fraction`, `enhance_for_vision` (autocontrast+gamma for faint plots). |
| `ocr.py` | `region_text`: PDF vector text in a region, else Tesseract OCR on the enhanced crop. Multimodal hint for Stage 2. |
| `vision_extractor.py` | OpenAI calls: `extract_structured` (schema-guaranteed), `extract_with_tools` (the slab tool-loop), `extract_from_image`. |
| `pdf_to_images.py` | Render pages / regions to PNG, region→mini-PDF helpers. |
| `make_region.py` | Interactive (tkinter) region picker → writes `input/regions/<pdf>.json`. |
| `main_1.py … main_9.py` | Pattern-specific extractors (vision tool-loop). **Left untouched.** |
| `config.py` | `INPUT_DIR`, `OUTPUT_DIR`, OpenAI key/model from `.env`. |
| `extraction_guard.py` | `ExtractionState`, record building, JSON cleanup for the tool-loop. |

---

## 4. The 9 slab patterns

Defined as **data** in `pattern_detector.SLAB_PATTERN_SIGNATURES` (columns +
required/forbidden feature flags). Stage 3 scores each signature against the
detected layout features and picks the best, or `None`.

| # | Shape |
|---|-------|
| 1 | TYPE / THICKNESS / STEEL ALONG SPAN / ACROSS SPAN / REMARKS |
| 2 | TYPE / THICKNESS / ALONG SHORT SPAN / ACROSS SHORT SPAN / REMARKS |
| 3 | SLAB MARKED / THK / TYPE / REINFORCEMENT(SHORT BAR, LONG BAR) / REMARKS |
| 4 | SLAB NO / THK / MIX / MAIN STEEL / EX. TOP OVER SUPPORT / DIST. STEEL |
| 5 | BAR MARKED / DIA / SPACING / REMARKS |
| 6 | BOTTOM/TOP SUPPORT, SHORT/LONG SPAN, with MAIN+DIST labels |
| 7 | BOTTOM/TOP SUPPORT, SHORT/LONG SPAN, no MAIN/DIST labels |
| 8 | NOS / THK / MAIN REINF / DISTRIBUTION REINF / REMARKS |
| 9 | BOTTOM/TOP SUPPORT each split into MAIN + DIST |

---

## 5. Guards (why bad output doesn't slip through)

- **Ink guard** (`image_enhance.ink_fraction`, background-relative): a near-empty
  crop is skipped instead of fed to the model. Stops the "blank image → fabricated
  S1..S100" failure. Light-gray content counts as ink (threshold is relative to the
  page background, not a fixed dark cutoff).
- **Fabrication post-check** (`auto_runner._looks_fabricated`): flags outputs that
  look hallucinated — many identical rows, or a perfect `S1..SN` sequence — and drops
  a `.SUSPECT.txt` marker.
- **Transposed detection**: ≥2 slab marks on one row → read as columns (Engine A) so
  a transposed table isn't mis-read as a single slab.
- **Low-confidence warning**: when Stage 3 isn't a clean rules match, the columns it
  saw are printed for review.
- **Decompression-bomb safety**: A0/A1 sheets at 400 DPI are ~1e9 px; Engine B renders
  bounded tiles and size-capped detection/crop images so PIL never chokes.

---

## 6. Faint / light-gray sheets (important)

Many CAD plots draw tables in light gray (~200–245 luminance), not black. A fixed
"dark < 160" rule treated them as blank and broke OpenCV detection. Everything that
looks at luminance now thresholds **relative to the page background** (99th
percentile ≈ paper white), so faint gray registers as content while true white does
not. Crops sent to the model are darkened by `enhance_for_vision` (autocontrast +
gamma — gentle, keeps thin reinforcement text readable; not harsh binarization).

Ultra-faint sheets (lines ~242 on a 255 background) are at the edge of automatic
detection — use a **region hint** for those.

---

## 7. Region hints & the picker

For any sheet auto-detection can't handle (ultra-faint, weird layout, no borders):

```
python src/make_region.py                      # pick a PDF from input/
python src/make_region.py "input/my sheet.pdf"
```

Drag a box around each slab schedule (preview is contrast-enhanced so faint tables
show), press **S**. It writes `input/regions/<pdf>.json` (normalized `x1,y1,x2,y2`),
which the router treats as a **guaranteed** hint (Engine: MANUAL REGION) — crop →
classify → extract, no guessing.

---

## 8. Output

```
output/<pdf>/detection.json              # what was found, which engine, regions
output/<pdf>__slab_<n>/<...>.json        # the extracted slabs for schedule n
```

Slab record schema:

```json
{
  "slab_id": "S2",
  "thickness": 175,
  "type": "TWO WAY",
  "mix": "M25",
  "reinforcement": { "dia": ["T10"], "spacing": ["100 C/C", "200 C/C"] }
}
```

---

## 9. Running

```
pip install -r requirements.txt          # + Tesseract system binary for OCR (optional)
# put PDFs in input/, set OPENAI_API_KEY in .env
python src/auto_runner.py
```

`pytesseract` is only the wrapper — install the **Tesseract** program separately
(Windows: UB-Mannheim build, add to PATH) for the OCR hint. Without it, the pipeline
runs image-only (no regression).

---

## 10. Extending

- **New pattern**: add a signature dict to `SLAB_PATTERN_SIGNATURES` (columns +
  required/forbidden flags) and a matching `main_<n>.py`. Stage 3 picks it up.
- **New title spelling**: already handled by the fuzzy regex; only touch
  `_TITLE_RE` for genuinely new wording.
- **Tune faint detection**: `table_locator` `ink_mask(delta=…)` (lower = more
  sensitive); Engine B detection render resolution in `sliding_window._capped_page_png`.
- **New reinforcement notation**: extend `_parse_reinforcement` in `vector_extractor`.

---

## 11. Known limits

- Ultra-faint, borderless tables → region hint.
- Vision classification (Stage 3 on an image) and `main_1..9` need the OpenAI API.
- Transposed parsing is tuned for the common "marks across the top, properties down"
  layout; exotic merged-header transposes may need tuning.
- Each scanned sheet may cost several Stage-2 calls (one per ruled candidate table);
  the classifier filters non-slab tables out.
