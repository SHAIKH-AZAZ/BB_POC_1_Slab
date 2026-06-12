"""
pattern_detector.py - Slab project

Hierarchical, layout-feature-driven detection.

    Stage 1  detect_document_family()   -> is this even a slab schedule?
    Stage 2  extract_layout_features()  -> structured facts about the page
    Stage 3  classify_slab_pattern()    -> deterministic Python rule matcher (1-9)
    -------  detect_document()          -> orchestrates 1->2->3
    -------  detect_pattern()           -> backward-compatible int wrapper

Why this shape (vs. the old "return ONLY a number 1-9" prompt):
  * The old detector was pattern-number driven. It FORCED every drawing into a
    slab pattern, even column / beam / footing schedules, because "not a slab
    schedule" was not an allowed answer.
  * Here the model only reports FACTS (family + layout features). Python decides
    the pattern from those facts, so classification is deterministic, debuggable
    and easy to extend (add a signature, not a 200-line prompt).
"""

import json
import os

from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image

# ===============================================================
# DOCUMENT FAMILIES
# ===============================================================

SLAB_FAMILY = "SLAB_SCHEDULE"

DOCUMENT_FAMILIES = [
    "SLAB_SCHEDULE",
    "COLUMN_SCHEDULE",
    "BEAM_SCHEDULE",
    "FOOTING_SCHEDULE",
    "WALL_SCHEDULE",
    "STAIR_SCHEDULE",
    "GENERAL_TABLE",
    "UNKNOWN",
]


# ===============================================================
# STAGE 1 - DOCUMENT FAMILY GATE
# ===============================================================

_FAMILY_PROMPT = """You are a structural drawing classifier.
Look at this engineering drawing/table and decide its PRIMARY schedule type.

Return ONLY valid JSON, no prose:
{
  "document_type": "<ONE of the allowed values>",
  "confidence": <number 0.0 - 1.0>,
  "reason": "<short phrase>"
}

Allowed document_type values:
  SLAB_SCHEDULE    - reinforcement schedule for floor/roof SLABS (slab marks like
                     S1, S2, SQ,S1M,STB1, thickness, steel ALONG/ACROSS span, top/bottom reinf).
  COLUMN_SCHEDULE  - column schedule (column marks C1/C2, floor levels, vertical
                     bars, ties/links, often with cross-section sketches).
  BEAM_SCHEDULE    - beam schedule (beam marks B1/B2, top/bottom bars, stirrups).
  FOOTING_SCHEDULE - footing/foundation schedule (footing marks, size LxBxD).
  WALL_SCHEDULE    - shear/retaining wall reinforcement schedule.
  STAIR_SCHEDULE   - staircase reinforcement schedule.
  GENERAL_TABLE    - a tabular drawing that is none of the above.
  UNKNOWN          - cannot tell.

Decide from STRUCTURE and HEADINGS, not from a single keyword.
Return ONLY the JSON object."""


def detect_document_family(image_path):
    """
    Stage 1. Returns dict: {document_type, confidence, reason}.
    Never raises on a well-formed-but-unexpected answer; falls back to UNKNOWN.
    """
    raw = extract_from_image(image_path, _FAMILY_PROMPT)
    data = _safe_json(raw)

    doc_type = str(data.get("document_type", "")).strip().upper()
    if doc_type not in DOCUMENT_FAMILIES:
        doc_type = "UNKNOWN"

    confidence = _safe_float(data.get("confidence"), default=0.0)
    reason = str(data.get("reason", "")).strip()

    return {
        "document_type": doc_type,
        "confidence": round(confidence, 3),
        "reason": reason,
    }


# ===============================================================
# STAGE 2 - LAYOUT FEATURE EXTRACTION
# ===============================================================

_FEATURE_FLAGS = [
    "has_table",
    "has_cross_sections",
    "has_elevations",
    "has_floor_labels",
    "has_type_groups",
    "has_column_marks",
    "mentions_along_across_span",  # "STEEL ALONG SPAN" / "ACROSS SPAN" (no short/long)
    "mentions_short_long_span",  # "SHORT SPAN" / "LONG SPAN"
    "mentions_main_dist_labels",  # "MAIN REINF." AND "DISTRIBUTION REINF." labelled
    "mentions_bottom_top_support",  # "BOTTOM REINFORCEMENT" + "TOP SUPPORT REINFORCEMENT"
    "mentions_mid_span",  # "IN MID SPAN"
    "mentions_short_long_bar",  # "SHORT BAR" / "LONG BAR" under a REINFORCEMENT group
    "mentions_main_steel_group",  # grouped "MAIN STEEL" header
    "mentions_ex_top_over_support",  # "EX. TOP OVER SUPPORT"
    "mentions_dia_spacing_columns",  # columns are essentially DIA + SPACING
    "mentions_bar_marked",  # a "BAR MARKED" / "BAR MARK" column
    "mentions_mix_column",  # a concrete MIX column
]

_FEATURE_PROMPT = """Analyze the LAYOUT of this slab reinforcement schedule.
Report only what you can actually see. Return ONLY valid JSON, no prose.

{
  "table_columns": ["<top-level column header texts, left to right>"],
  "header_keywords": ["<distinct words appearing in the header area>"],
  "has_table": true,
  "has_cross_sections": false,
  "has_elevations": false,
  "has_floor_labels": false,
  "has_type_groups": false,
  "has_column_marks": false,
  "mentions_along_across_span": false,
  "mentions_short_long_span": false,
  "mentions_main_dist_labels": false,
  "mentions_bottom_top_support": false,
  "mentions_mid_span": false,
  "mentions_short_long_bar": false,
  "mentions_main_steel_group": false,
  "mentions_ex_top_over_support": false,
  "mentions_dia_spacing_columns": false,
  "mentions_bar_marked": false,
  "mentions_mix_column": false
}

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

Return ONLY the JSON object."""


def extract_layout_features(image_path):
    """
    Stage 2. Returns a normalized feature dict. Missing keys are coerced to
    safe defaults so Stage 3 can rely on every flag existing.
    """
    raw = extract_from_image(image_path, _FEATURE_PROMPT)
    data = _safe_json(raw)

    features = {flag: _safe_bool(data.get(flag)) for flag in _FEATURE_FLAGS}
    features["table_columns"] = _as_str_list(data.get("table_columns"))
    features["header_keywords"] = _as_str_list(data.get("header_keywords"))
    return features


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
    "OF",
    "TO",
    "THE",
    "AND",
    "FOR",
    "IN",
    "ON",
    "AT",
    "BE",
    "WITH",
    "A",
    "AN",
    "&",
    "-",
    "/",
    "REINF",
    "REINFORCEMENT",
    "REMARKS",
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


def detect_document(pdf_path, temp_folder, family_threshold=0.0):
    """
    Full hierarchical detection. Returns:
        {
          "document_type": <family>,
          "family_confidence": float,
          "pattern": <int 1-9 or None>,
          "pattern_confidence": float,
          "method": str,
          "features": {...} | None,
          "image_path": str
        }

    Only SLAB_SCHEDULE documents proceed to Stage 2/3. Anything else returns
    early with pattern=None instead of being force-fit into a slab pattern.
    """
    image_paths = convert_pdf_to_images(pdf_path, temp_folder)
    if not image_paths:
        raise Exception("No image generated for pattern detection.")
    first_image = image_paths[0]

    family = detect_document_family(first_image)
    result = {
        "document_type": family["document_type"],
        "family_confidence": family["confidence"],
        "family_reason": family.get("reason", ""),
        "pattern": None,
        "pattern_confidence": 0.0,
        "method": "family_gate",
        "matched_label": None,
        "features": None,
        "image_path": first_image,
    }

    if family["document_type"] != SLAB_FAMILY:
        return result

    features = extract_layout_features(first_image)
    classification = classify_slab_pattern(features)

    result.update(
        {
            "pattern": classification["pattern"],
            "pattern_confidence": classification["confidence"],
            "method": classification["method"],
            "matched_label": classification["matched_label"],
            "features": features,
            "pattern_scores": classification["scores"],
        }
    )
    return result


def detect_pattern(pdf_path, temp_folder):
    """
    Backward-compatible wrapper. Returns an int pattern (1-9) for slab schedules.

    Raises a descriptive exception when the drawing is not a slab schedule or the
    pattern cannot be determined, so old call sites fail loudly instead of acting
    on a wrong number. New code should prefer detect_document().
    """
    result = detect_document(pdf_path, temp_folder)

    if result["document_type"] != SLAB_FAMILY:
        raise Exception(
            f"Not a slab schedule (detected {result['document_type']}, "
            f"confidence {result['family_confidence']}). No slab pattern assigned."
        )

    if result["pattern"] is None:
        raise Exception(
            "Document is a slab schedule but no pattern (1-9) matched its layout. "
            f"Features: {json.dumps(result.get('features'))}"
        )

    return int(result["pattern"])


# ===============================================================
# HELPERS
# ===============================================================


def _safe_json(raw):
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw).strip())
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return False


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(value).strip()]


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
