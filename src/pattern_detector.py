"""
pattern_detector.py - Slab project

Hierarchical, layout-feature-driven, MULTI-SCHEDULE detection.

    Stage 1  detect_schedules()         -> find EVERY schedule on the sheet
                                           (type + title + bounding box)
    Stage 2  extract_layout_features()  -> structured facts about a slab table
    Stage 3  classify_slab_pattern()    -> deterministic Python rule matcher (1-9)
    -------  detect_document()          -> orchestrates 1 -> crop -> 2 -> 3
    -------  detect_pattern()           -> backward-compatible int wrapper

Why this shape:
  * Old detector: "return ONLY a number 1-9" -> forced every drawing into a slab
    pattern, even beam/column sheets.
  * v1 fix: a single family gate -> but real RCC sheets carry MANY schedules
    (beam + slab + column) at once, so a slab table got skipped when beams
    dominated the page.
  * v2 (this file): Stage 1 detects ALL schedules and their locations. Each slab
    schedule is cropped into its own region and classified/extracted on its own,
    so a slab table is never lost just because beams share the sheet.

Stages 1 and 2 use schema-guaranteed structured outputs (Pydantic), so there is
no fragile hand JSON parsing on the happy path.
"""

import json
import os
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from pdf_to_images import convert_pdf_to_images, crop_region_to_pdf
from vision_extractor import extract_structured
from table_locator import snap_region


# ===============================================================
# DOCUMENT FAMILIES
# ===============================================================

SLAB_FAMILY = "SLAB_SCHEDULE"


class DocumentType(str, Enum):
    SLAB_SCHEDULE = "SLAB_SCHEDULE"
    COLUMN_SCHEDULE = "COLUMN_SCHEDULE"
    BEAM_SCHEDULE = "BEAM_SCHEDULE"
    FOOTING_SCHEDULE = "FOOTING_SCHEDULE"
    WALL_SCHEDULE = "WALL_SCHEDULE"
    STAIR_SCHEDULE = "STAIR_SCHEDULE"
    GENERAL_TABLE = "GENERAL_TABLE"
    UNKNOWN = "UNKNOWN"


DOCUMENT_FAMILIES = [d.value for d in DocumentType]


# ===============================================================
# STAGE 1 - MULTI-SCHEDULE DETECTION + LOCALIZATION
# ===============================================================

class ScheduleRegion(BaseModel):
    """One detected schedule table on the sheet, with its location."""
    document_type: DocumentType = Field(description="Family of this schedule.")
    title: str = Field(description="Exact title/heading text above the table.")
    confidence: float = Field(description="0.0-1.0 confidence in document_type.")
    x1: float = Field(description="Left edge of the table, 0.0-1.0 of page width.")
    y1: float = Field(description="Top edge of the table, 0.0-1.0 of page height.")
    x2: float = Field(description="Right edge of the table, 0.0-1.0 of page width.")
    y2: float = Field(description="Bottom edge of the table, 0.0-1.0 of page height.")


class SheetSchedules(BaseModel):
    schedules: List[ScheduleRegion]


_SHEET_PROMPT = """You are a structural drawing analyst.
This is ONE engineering sheet that may contain SEVERAL separate schedule tables
(for example a beam schedule AND a slab schedule AND a column schedule).

Find EVERY distinct schedule TABLE on the sheet. For each one report:
  - document_type  (see allowed values)
  - title          (the exact heading printed above the table)
  - confidence     (0.0 - 1.0)
  - x1, y1, x2, y2 : the table's bounding box as FRACTIONS of the full page
                     (0,0 = top-left, 1,1 = bottom-right). The box must enclose
                     the WHOLE table: its title, header row and all data rows.

Allowed document_type values:
  SLAB_SCHEDULE    - reinforcement schedule for floor/roof SLABS (slab marks like
                     S1, S2, SQ, S1M, STB1; thickness; steel ALONG/ACROSS span;
                     SHORT BAR / LONG BAR; top/bottom reinforcement).
  COLUMN_SCHEDULE  - column schedule (column marks C1/C2, floor levels, vertical
                     bars, ties/links, cross-section sketches).
  BEAM_SCHEDULE    - beam schedule (beam marks B1/B2, top/bottom bars, stirrups).
  FOOTING_SCHEDULE - footing/foundation schedule (footing marks, size LxBxD).
  WALL_SCHEDULE    - shear/retaining wall reinforcement schedule.
  STAIR_SCHEDULE   - staircase reinforcement schedule.
  GENERAL_TABLE    - a table that is none of the above.
  UNKNOWN          - cannot tell.

Rules:
  - List a schedule even if it is small or in a corner. Do NOT report only the
    largest one.
  - A schedule is a TABLE with a header row and multiple data ROWS (e.g. S1, S2,
    S3 ...). It is NOT a drawing.
  - A reinforcement LAYOUT / PLAN drawing — bars drawn on a floor plan, slab
    panels with rebar callouts, sections, or detail sketches — is NOT a schedule.
    If the sheet's "slab" content is a layout drawing rather than a row/column
    table, do NOT report it as SLAB_SCHEDULE (use GENERAL_TABLE or omit it).
  - Only report SLAB_SCHEDULE when you can actually see a gridded table whose rows
    are individual slab marks with columns of values.
  - If the same schedule continues in two stacked blocks, report each block."""


def detect_schedules(image_path):
    """
    Stage 1. Returns a list of dicts, one per schedule table found on the sheet:
        [{document_type, title, confidence, region:{x1,y1,x2,y2}}, ...]
    """
    parsed = extract_structured(image_path, _SHEET_PROMPT, SheetSchedules)

    out = []
    for s in parsed.schedules:
        out.append({
            "document_type": s.document_type.value,
            "title": (s.title or "").strip(),
            "confidence": round(float(s.confidence), 3),
            "region": _region_dict(s.x1, s.y1, s.x2, s.y2),
        })
    return out


def detect_document_family(image_path):
    """
    Convenience: the single DOMINANT schedule family on the sheet (largest-area,
    confidence-weighted). Kept for callers that want one label. Returns
    {document_type, confidence, reason}.
    """
    return _dominant(detect_schedules(image_path))


# ===============================================================
# STAGE 2 - LAYOUT FEATURE EXTRACTION
# ===============================================================

class LayoutFeatures(BaseModel):
    table_columns: List[str] = Field(description="Top-level column headers, left to right.")
    header_keywords: List[str] = Field(description="Distinct words in the header area.")
    has_table: bool
    has_cross_sections: bool
    has_elevations: bool
    has_floor_labels: bool
    has_type_groups: bool
    has_column_marks: bool
    mentions_along_across_span: bool        # "STEEL ALONG SPAN" / "ACROSS SPAN" (no short/long)
    mentions_short_long_span: bool          # "SHORT SPAN" / "LONG SPAN"
    mentions_main_dist_labels: bool         # "MAIN REINF." AND "DISTRIBUTION REINF."
    mentions_bottom_top_support: bool       # "BOTTOM REINFORCEMENT" + "TOP SUPPORT REINFORCEMENT"
    mentions_mid_span: bool                 # "IN MID SPAN"
    mentions_short_long_bar: bool           # "SHORT BAR" / "LONG BAR"
    mentions_main_steel_group: bool         # grouped "MAIN STEEL" header
    mentions_ex_top_over_support: bool      # "EX. TOP OVER SUPPORT"
    mentions_dia_spacing_columns: bool      # columns are essentially DIA + SPACING
    mentions_bar_marked: bool               # a "BAR MARKED" / "BAR MARK" column
    mentions_mix_column: bool               # a concrete MIX column


# Flags used by the Stage 3 rule matcher (kept as a list for iteration / tests).
_FEATURE_FLAGS = [
    name for name in LayoutFeatures.model_fields
    if name not in ("table_columns", "header_keywords")
]


_FEATURE_PROMPT = """Analyze the LAYOUT of this slab reinforcement schedule table.

CRITICAL: Report ONLY columns and words that are LITERALLY PRINTED in this image.
Do NOT guess a "typical" slab schedule. Do NOT invent columns such as DIRECTION,
BOTTOM REINFORCEMENT or TOP SUPPORT REINFORCEMENT unless those exact words appear.
Read the header text left to right and transcribe it verbatim. If a flag's words
are not visibly present, that flag MUST be false.

Flag meanings:
  mentions_along_across_span    -> header says "ALONG SPAN" / "ACROSS SPAN" (no short/long).
  mentions_short_long_span      -> header says "SHORT SPAN" and/or "LONG SPAN".
  mentions_main_dist_labels     -> "MAIN REINF." AND "DISTRIBUTION REINF." are explicit labels.
  mentions_bottom_top_support   -> has BOTH "BOTTOM REINFORCEMENT" and "TOP SUPPORT REINFORCEMENT".
  mentions_mid_span             -> "IN MID SPAN" appears.
  mentions_short_long_bar       -> a REINFORCEMENT group split into "SHORT BAR" / "LONG BAR".
  mentions_main_steel_group     -> a grouped "MAIN STEEL" header.
  mentions_ex_top_over_support  -> "EX. TOP OVER SUPPORT".
  mentions_dia_spacing_columns  -> the table is basically DIA + SPACING columns.
  mentions_bar_marked           -> a "BAR MARKED" / "BAR MARK" column.
  mentions_mix_column           -> a concrete MIX/grade column.

Set table_columns to the top-level column headers left-to-right, and
header_keywords to the distinct words you see in the header area."""


def extract_layout_features(image_path):
    """
    Stage 2. Returns a plain dict (every flag guaranteed present) so the Stage 3
    matcher can rely on it. Uses schema-guaranteed structured output.
    """
    parsed = extract_structured(image_path, _FEATURE_PROMPT, LayoutFeatures)
    return parsed.model_dump()


# ===============================================================
# STAGE 3 - DETERMINISTIC PATTERN RULE MATCHER
# ===============================================================
#
# Pattern signatures are DATA, not prose. Each signature lists:
#   columns   : canonical top-level column headers (used for fuzzy column scoring)
#   required  : feature flags that should be True for this pattern
#   forbidden : feature flags that must be False for this pattern
#
# The matcher prefers signatures whose required/forbidden constraints are all
# satisfied, then breaks ties by column-header similarity. If nothing satisfies
# the hard constraints it falls back to pure column similarity, so an OCR miss on
# one flag degrades gracefully instead of crashing.

SLAB_PATTERN_SIGNATURES = [
    {
        "pattern": 1,
        "label": "TYPE / THICKNESS / ALONG SPAN / ACROSS SPAN",
        "columns": [
            "TYPE",
            "THICKNESS",
            "STEEL ALONG SPAN",
            "STEEL ACROSS SPAN",
            "REMARKS",
        ],
        "required": ["mentions_along_across_span"],
        "forbidden": [
            "mentions_short_long_span",
            "mentions_bottom_top_support",
            "mentions_main_dist_labels",
            "mentions_short_long_bar",
            "mentions_main_steel_group",
            "mentions_dia_spacing_columns",
        ],
    },
    {
        "pattern": 2,
        "label": "TYPE / THICKNESS / ALONG SHORT SPAN / ACROSS SHORT SPAN",
        "columns": [
            "TYPE",
            "THICKNESS",
            "STEEL ALONG SHORT SPAN",
            "STEEL ACROSS SHORT SPAN",
            "REMARKS",
        ],
        "required": ["mentions_short_long_span"],
        "forbidden": [
            "mentions_bottom_top_support",
            "mentions_main_dist_labels",
            "mentions_short_long_bar",
            "mentions_main_steel_group",
            "mentions_mid_span",
            "mentions_dia_spacing_columns",
        ],
    },
    {
        "pattern": 3,
        "label": "SLAB MARKED / THK / TYPE / REINFORCEMENT(SHORT BAR, LONG BAR)",
        "columns": [
            "SLAB MARKED",
            "THK",
            "TYPE",
            "REINFORCEMENT",
            "SHORT BAR",
            "LONG BAR",
            "REMARKS",
        ],
        "required": ["mentions_short_long_bar"],
        "forbidden": [
            "mentions_bottom_top_support",
            "mentions_main_steel_group",
            "mentions_ex_top_over_support",
        ],
    },
    {
        "pattern": 4,
        "label": "SLAB NO / THK / MIX / MAIN STEEL / EX. TOP OVER SUPPORT / DIST. STEEL",
        "columns": [
            "SLAB NO",
            "SLAB THK",
            "MIX",
            "TYPE",
            "MAIN STEEL",
            "EX TOP OVER SUPPORT",
            "DIST STEEL",
        ],
        "required": ["mentions_main_steel_group", "mentions_ex_top_over_support"],
        "forbidden": ["mentions_bottom_top_support"],
    },
    {
        "pattern": 5,
        "label": "BAR MARKED / DIA / SPACING",
        "columns": ["BAR MARKED", "DIA", "SPACING", "REMARKS"],
        "required": ["mentions_dia_spacing_columns"],
        "forbidden": [
            "mentions_bottom_top_support",
            "mentions_main_steel_group",
            "mentions_short_long_bar",
            "mentions_along_across_span",
        ],
    },
    {
        "pattern": 6,
        "label": "BOTTOM/TOP SUPPORT, SHORT/LONG SPAN, with MAIN+DIST labels",
        "columns": [
            "SLAB MARKED",
            "SLAB THICKNESS",
            "BOTTOM REINFORCEMENT ALONG SHORT SPAN MAIN REINF",
            "BOTTOM REINFORCEMENT ALONG LONG SPAN DISTRIBUTION REINF",
            "TOP SUPPORT REINFORCEMENT ALONG SHORT SPAN",
            "TOP SUPPORT REINFORCEMENT ALONG LONG SPAN",
            "IN MID SPAN",
            "DISTRIBUTION",
            "REMARKS",
        ],
        "required": [
            "mentions_bottom_top_support",
            "mentions_short_long_span",
            "mentions_main_dist_labels",
        ],
        "forbidden": [],
    },
    {
        "pattern": 7,
        "label": "BOTTOM/TOP SUPPORT, SHORT/LONG SPAN, NO MAIN/DIST labels",
        "columns": [
            "SLAB MARKED",
            "SLAB THICKNESS",
            "BOTTOM REINFORCEMENT ALONG SHORT SPAN",
            "BOTTOM REINFORCEMENT ALONG LONG SPAN",
            "TOP SUPPORT REINFORCEMENT ALONG SHORT SPAN",
            "TOP SUPPORT REINFORCEMENT ALONG LONG SPAN",
            "IN MID SPAN",
            "DISTRIBUTION",
            "REMARKS",
        ],
        "required": ["mentions_bottom_top_support", "mentions_short_long_span"],
        "forbidden": ["mentions_main_dist_labels"],
    },
    {
        "pattern": 8,
        "label": "NOS / THK / MAIN REINF / DISTRIBUTION REINF",
        "columns": ["NOS", "THK", "MAIN REINF", "DISTRIBUTION REINF", "REMARKS"],
        "required": ["mentions_main_dist_labels"],
        "forbidden": [
            "mentions_bottom_top_support",
            "mentions_short_long_span",
            "mentions_short_long_bar",
            "mentions_main_steel_group",
        ],
    },
    {
        "pattern": 9,
        "label": "BOTTOM/TOP SUPPORT each split into MAIN + DIST",
        "columns": [
            "SLAB MARKED",
            "SLAB THICKNESS",
            "BOTTOM REINFORCEMENT MAIN REINF",
            "BOTTOM REINFORCEMENT DISTRIBUTION REINF",
            "TOP SUPPORT REINFORCEMENT MAIN REINF",
            "TOP SUPPORT REINFORCEMENT DISTRIBUTION REINF",
            "IN MID SPAN",
            "DISTRIBUTION",
            "REMARKS",
        ],
        "required": ["mentions_bottom_top_support", "mentions_main_dist_labels"],
        "forbidden": ["mentions_short_long_span"],
    },
]

# Words that carry no discriminating power for column similarity.
_STOPWORDS = {
    "OF", "TO", "THE", "AND", "FOR", "IN", "ON", "AT", "BE", "WITH",
    "A", "AN", "&", "-", "/", "REINF", "REINFORCEMENT", "REMARKS",
}


def classify_slab_pattern(features):
    """
    Stage 3. Deterministic. Returns dict:
        {
          "pattern": <int 1-9 or None>,
          "confidence": <0.0-1.0>,
          "method": "rules" | "columns" | "none",
          "scores": [ {pattern, hard_ok, column_score, flag_score, total}, ... ]
        }
    """
    observed_tokens = _tokenize(
        features.get("table_columns", []) + features.get("header_keywords", [])
    )

    scored = []
    for sig in SLAB_PATTERN_SIGNATURES:
        required = sig.get("required", [])
        forbidden = sig.get("forbidden", [])

        req_hits = sum(1 for f in required if features.get(f))
        forb_hits = sum(1 for f in forbidden if features.get(f))

        hard_ok = (req_hits == len(required)) and (forb_hits == 0)

        sig_tokens = _tokenize(sig["columns"])
        column_score = _jaccard(observed_tokens, sig_tokens)

        # flag_score rewards satisfied requireds and penalizes tripped forbiddens
        flag_score = req_hits - forb_hits

        scored.append(
            {
                "pattern": sig["pattern"],
                "label": sig["label"],
                "hard_ok": hard_ok,
                "req_hits": req_hits,
                "req_total": len(required),
                "forb_hits": forb_hits,
                "column_score": round(column_score, 3),
                "flag_score": flag_score,
                "total": round(flag_score + column_score, 3),
            }
        )

    # Prefer signatures that satisfy hard constraints; rank by total score.
    hard_candidates = [s for s in scored if s["hard_ok"]]

    if hard_candidates:
        best = max(hard_candidates, key=lambda s: (s["total"], s["column_score"]))
        method = "rules"
    else:
        best = max(scored, key=lambda s: (s["column_score"], s["flag_score"]))
        method = "columns"

    # Confidence: high when a hard rule wins clearly; otherwise reflect column fit.
    if method == "rules":
        runner_up = max(
            (s["total"] for s in hard_candidates if s["pattern"] != best["pattern"]),
            default=0.0,
        )
        confidence = min(1.0, 0.6 + 0.4 * max(0.0, best["total"] - runner_up) / 3.0)
    else:
        confidence = round(0.4 * best["column_score"], 3)

    pattern = (
        best["pattern"] if (method == "rules" or best["column_score"] >= 0.34) else None
    )
    if pattern is None:
        method = "none"

    return {
        "pattern": pattern,
        "confidence": round(confidence, 3),
        "method": method,
        "matched_label": best["label"] if pattern else None,
        "scores": sorted(scored, key=lambda s: s["total"], reverse=True),
    }


# ===============================================================
# ORCHESTRATION
# ===============================================================

def detect_document(pdf_path, temp_folder, slab_pad=0.03):
    """
    Full multi-schedule detection. Returns:
        {
          "schedules": [ {document_type, title, confidence, region}, ... ],
          "document_type": <dominant family>,
          "family_confidence": float,
          "slab_targets": [
              {
                "index": int, "title": str, "confidence": float,
                "region": {...},
                "image_path": <cropped PNG>,   # for inspection / re-detection
                "pdf_path":  <slab-only PDF>,   # feed to main_1..9 unchanged
                "pattern": <int 1-9 or None>,
                "pattern_confidence": float,
                "method": str,
                "features": {...},
                "pattern_scores": [...]
              }, ...
          ],
          "page_image": <full page PNG>,
          "pdf_path": <original PDF>
        }

    Every SLAB_SCHEDULE on the sheet becomes its own target (cropped + classified),
    so a slab table is never dropped because beams/columns share the page.
    """
    image_paths = convert_pdf_to_images(pdf_path, temp_folder)
    if not image_paths:
        raise Exception("No image generated for detection.")
    page_image = image_paths[0]

    schedules = detect_schedules(page_image)
    dominant = _dominant(schedules)

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    slab_targets = []

    slab_schedules = [s for s in schedules if s["document_type"] == SLAB_FAMILY]
    for idx, s in enumerate(slab_schedules, start=1):
        crop_name = f"{base}__slab_{idx}"

        # Snap the model's approximate box to the real ruled-table rectangle so the
        # crop captures EVERY row. Fall back to a generously padded model box if no
        # ruled table is found (e.g. borderless/scanned tables, or no OpenCV).
        snapped = snap_region(page_image, s["region"])
        crop_region = snapped if snapped else s["region"]
        crop_pad = 0.006 if snapped else max(slab_pad, 0.05)

        crop_png, crop_pdf = crop_region_to_pdf(
            pdf_path, crop_region, crop_name, temp_folder, pad=crop_pad
        )

        # GUARD: a blank / near-empty crop means the model pointed at a layout
        # drawing or empty strip, not a real schedule table. Extracting from it
        # makes the vision model fabricate rows (e.g. S1..S100 cloned). Skip it.
        ink = _ink_fraction(crop_png)
        valid = (ink is None) or (ink >= MIN_TABLE_INK)

        if not valid:
            slab_targets.append({
                "index": idx,
                "title": s["title"],
                "confidence": s["confidence"],
                "model_region": s["region"],
                "region": crop_region,
                "snapped_to_table": bool(snapped),
                "image_path": crop_png,
                "pdf_path": crop_pdf,
                "ink_fraction": ink,
                "valid": False,
                "skip_reason": (
                    f"crop is near-empty (ink={ink}); likely a layout drawing or "
                    "mislocated region, not a real schedule table"
                ),
                "pattern": None,
                "pattern_confidence": 0.0,
                "method": "skipped_blank",
                "matched_label": None,
                "features": None,
                "pattern_scores": [],
            })
            continue

        features = extract_layout_features(crop_png)
        classification = classify_slab_pattern(features)

        # GUARD: Stage 2 must actually see a table; otherwise treat as not-a-schedule.
        if not features.get("has_table", True):
            classification = {"pattern": None, "confidence": 0.0,
                              "method": "no_table", "matched_label": None, "scores": []}

        slab_targets.append({
            "index": idx,
            "title": s["title"],
            "confidence": s["confidence"],
            "model_region": s["region"],
            "region": crop_region,
            "snapped_to_table": bool(snapped),
            "image_path": crop_png,
            "pdf_path": crop_pdf,
            "ink_fraction": ink,
            "valid": True,
            "skip_reason": None,
            "pattern": classification["pattern"],
            "pattern_confidence": classification["confidence"],
            "method": classification["method"],
            "matched_label": classification["matched_label"],
            "features": features,
            "pattern_scores": classification["scores"],
        })

    return {
        "schedules": schedules,
        "document_type": dominant["document_type"],
        "family_confidence": dominant["confidence"],
        "slab_targets": slab_targets,
        "page_image": page_image,
        "pdf_path": pdf_path,
    }


def detect_pattern(pdf_path, temp_folder):
    """
    Backward-compatible wrapper. Returns the int pattern (1-9) of the FIRST slab
    schedule found on the sheet.

    Raises a descriptive exception when there is no slab schedule or no pattern
    matched, so old call sites fail loudly instead of acting on a wrong number.
    New code should prefer detect_document() (it returns every slab target).
    """
    result = detect_document(pdf_path, temp_folder)
    targets = result["slab_targets"]

    if not targets:
        raise Exception(
            f"No slab schedule found on the sheet (dominant: {result['document_type']}). "
            f"Schedules detected: {[s['document_type'] for s in result['schedules']]}"
        )

    first = targets[0]
    if first["pattern"] is None:
        raise Exception(
            "Slab schedule found but no pattern (1-9) matched its layout. "
            f"Features: {json.dumps(first.get('features'))}"
        )
    return int(first["pattern"])


# ===============================================================
# HELPERS
# ===============================================================

def _dominant(schedules):
    """Pick the single most prominent schedule (area x confidence)."""
    if not schedules:
        return {"document_type": "UNKNOWN", "confidence": 0.0, "reason": ""}

    def area(s):
        r = s["region"]
        return max(0.0, r["x2"] - r["x1"]) * max(0.0, r["y2"] - r["y1"])

    d = max(schedules, key=lambda s: area(s) * (0.5 + 0.5 * s["confidence"]))
    return {"document_type": d["document_type"], "confidence": d["confidence"],
            "reason": d["title"]}


def _region_dict(x1, y1, x2, y2):
    x1, x2 = sorted((_clamp01(x1), _clamp01(x2)))
    y1, y2 = sorted((_clamp01(y1), _clamp01(y2)))
    # guard against a degenerate / missing box -> treat as whole page
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        return {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    return {"x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4)}


def _clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _tokenize(strings):
    """Flatten a list of header strings into a set of significant uppercase words."""
    tokens = set()
    for s in strings:
        for word in (
            str(s).upper().replace("/", " ").replace("-", " ").replace(".", " ").split()
        ):
            word = "".join(ch for ch in word if ch.isalnum())
            if word and word not in _STOPWORDS:
                tokens.add(word)
    return tokens


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# A real schedule-table crop has several percent dark pixels; a blank/near-empty
# crop is ~0. Anything below this is treated as "no table here".
MIN_TABLE_INK = 0.01


def _ink_fraction(image_path):
    """
    Fraction of dark (ink) pixels in an image, 0.0-1.0. Uses PIL so it works even
    without OpenCV. Returns None if the image can't be read (guard then no-ops).
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        # downsample for speed on large crops
        img.thumbnail((1000, 1000))
        hist = img.histogram()          # 256 luminance bins
        dark = sum(hist[:160])          # pixels darker than 160 = ink
        total = sum(hist)
        return round(dark / total, 4) if total else None
    except Exception:
        return None
