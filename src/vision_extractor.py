"""
vision_extractor.py - Slab project

Tool-augmented extraction with enforced sequence:
think -> zoom_region -> confirm_read -> add_slab.

Image utilities available:
  strip_borders(path)              - remove whitespace margins
  slice_horizontal(path, n)        - cut image into n horizontal bands
  resize_image(path, max_side)     - scale longest side to max_side px
  crop_region(path, x1,y1,x2,y2)  - normalized-coord crop (0.0-1.0)
"""

import base64
import io
import json
import os
import time

from PIL import Image, ImageChops
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_IMAGE_DETAIL, OPENAI_MODEL
from extraction_guard import ExtractionState, build_slab_record, clean_json_string


client = OpenAI(api_key=OPENAI_API_KEY)


# ===============================================================
# IMAGE UTILITIES
# ===============================================================

def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def encode_pil(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_content(base64_image, detail=OPENAI_IMAGE_DETAIL):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": detail,
        },
    }


def strip_borders(image_path, threshold=240):
    """
    Remove near-white border margins from a scanned drawing image.
    Returns a PIL Image with whitespace cropped off all four edges.
    threshold: pixels brighter than this value on all channels are treated as background.
    """
    img = Image.open(image_path).convert("RGB")
    bg = Image.new("RGB", img.size, (threshold, threshold, threshold))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def slice_horizontal(image_path, n_slices=2):
    """
    Cut the image into n_slices equal horizontal bands.
    Returns a list of PIL Images (top to bottom).
    Useful when a slab table spans the full page height and the model
    misses rows near the bottom.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    band_h = h // n_slices
    slices = []
    for i in range(n_slices):
        top = i * band_h
        bottom = h if i == n_slices - 1 else (i + 1) * band_h
        slices.append(img.crop((0, top, w, bottom)))
    return slices


def slice_vertical(image_path, n_slices=2):
    """
    Cut the image into n_slices equal vertical bands (left to right).
    Returns a list of PIL Images.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    band_w = w // n_slices
    slices = []
    for i in range(n_slices):
        left = i * band_w
        right = w if i == n_slices - 1 else (i + 1) * band_w
        slices.append(img.crop((left, 0, right, h)))
    return slices


def resize_image(image_path, max_side=2000):
    """
    Scale image so its longest side is at most max_side pixels.
    Returns a PIL Image.
    """
    img = Image.open(image_path).convert("RGB")
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def crop_region(image_path, x1, y1, x2, y2, min_side=1200):
    """
    Crop a normalized-coordinate (0.0-1.0) region and upscale if small.
    Returns base64-encoded PNG string ready for the API.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1 = max(0.0, min(1.0, float(x1)))
    y1 = max(0.0, min(1.0, float(y1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y2 = max(0.0, min(1.0, float(y2)))
    if x2 <= x1:
        x2 = min(1.0, x1 + 0.05)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.05)
    left   = int(x1 * w)
    top    = int(y1 * h)
    right  = max(int(x2 * w), left + 20)
    bottom = max(int(y2 * h), top + 20)
    cropped = img.crop((left, top, right, bottom))
    longest = max(cropped.size)
    if longest < min_side:
        scale = min_side / longest
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)),
            Image.LANCZOS,
        )
    return encode_pil(cropped)


# Keep old name as alias so existing callers don't break
_crop_image_b64 = crop_region


# ===============================================================
# TOOL PROTOCOL PREAMBLE
# ===============================================================

_REGION_PURPOSE_ENUM = [
    "header",
    "row_label",
    "data_cell",
    "global_note",
    "ambiguous_text",
]


def _with_tool_protocol(prompt_text):
    return (
        "ENFORCED TOOL PROTOCOL:\n"
        "You MUST use tools — do NOT return raw JSON text.\n\n"

        "MIX EXTRACTION RULE:\n"
        "Slab tables may have a MIX column or a global mix note (e.g. M20, M25).\n"
        "  - A combined label like 'M25 : Fe500' → M25 is concrete mix, Fe500 is steel grade.\n"
        "  - Extract BOTH separately. Never ignore M-grade values.\n\n"

        "Step 1: call think() — identify all slab IDs and plan extraction.\n"
        "Step 2: call add_slab() once for EVERY slab row (top to bottom).\n"
        "  - Do NOT stop early. Every visible slab row must get its own add_slab call.\n"
        "  - steel_along_span and steel_across_span MUST contain the FULL cell text\n"
        "    (e.g. 'T12 @ 100 C/C ALT BENT UP'), not just the diameter token.\n"
        "Optional: call zoom_region() + confirm_read() for any cell that is blurry or ambiguous.\n\n"
        f"{prompt_text}"
    )


# ===============================================================
# SLAB TOOLS
# ===============================================================

SLAB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Mandatory first call. Return structured slab schedule planning data: "
                "table columns, all slab IDs, deleted rows, expected count, zoom targets, "
                "and all concrete/steel grades found anywhere in the table."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_quality": {"type": "string"},
                    "table_bounds": {"type": "object"},
                    "table_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Visible schedule columns, e.g. TYPE, THICKNESS, ALONG SPAN, ACROSS SPAN.",
                    },
                    "slab_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All slab IDs visible in the TYPE/SLAB MARKED column.",
                    },
                    "deleted_rows": {"type": "array", "items": {"type": "string"}},
                    "multi_line_cells": {"type": "array", "items": {"type": "string"}},
                    "concrete_grades": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ALL concrete mix grades (M-grades) found anywhere in the table — "
                            "in column headers, row cells, vertical labels, or notes. "
                            "A label like 'M25 : Fe500' → M25 is a concrete grade. "
                            "List ALL unique values, e.g. ['M25', 'M20']."
                        ),
                    },
                    "steel_grades": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "ALL steel grades (Fe-grades) found anywhere in the table. "
                            "A label like 'M25 : Fe500' → Fe500 is the steel grade. "
                            "List ALL unique values, e.g. ['Fe500']."
                        ),
                    },
                    "expected_count": {"type": "integer"},
                    "zoom_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "region_id": {"type": "string"},
                                "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                                "target": {"type": "string"},
                                "x1": {"type": "number", "description": "Left edge 0.0-1.0"},
                                "y1": {"type": "number", "description": "Top edge 0.0-1.0"},
                                "x2": {"type": "number", "description": "Right edge 0.0-1.0"},
                                "y2": {"type": "number", "description": "Bottom edge 0.0-1.0"},
                                "reason": {"type": "string"},
                            },
                            "required": ["region_id", "purpose", "target", "reason"],
                        },
                    },
                    "normalization_rules": {"type": "array", "items": {"type": "string"}},
                    "extraction_order": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "image_quality",
                    "table_bounds",
                    "table_columns",
                    "slab_ids",
                    "deleted_rows",
                    "multi_line_cells",
                    "concrete_grades",
                    "steel_grades",
                    "expected_count",
                    "zoom_plan",
                    "normalization_rules",
                    "extraction_order",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_region",
            "description": (
                "Crop and zoom a planned slab schedule region after think. "
                "Coordinates are NORMALIZED FRACTIONS of the image in the range 0.0 to 1.0. "
                "(0,0) = top-left, (1,1) = bottom-right. NOT pixels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                    "x1": {"type": "number", "description": "Left edge 0.0-1.0"},
                    "y1": {"type": "number", "description": "Top edge 0.0-1.0"},
                    "x2": {"type": "number", "description": "Right edge 0.0-1.0 (must be > x1)"},
                    "y2": {"type": "number", "description": "Bottom edge 0.0-1.0 (must be > y1)"},
                    "reason": {"type": "string"},
                },
                "required": ["region_id", "purpose", "x1", "y1", "x2", "y2", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_read",
            "description": "Record exact text read from a zoomed region before add_slab references it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "text": {"type": "string"},
                    "confidence": {"type": "string"},
                    "applies_to": {"type": "string"},
                },
                "required": ["region_id", "text", "confidence", "applies_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_slab",
            "description": (
                "Record ONE slab row. Call once per slab ID (top to bottom). "
                "Do NOT batch multiple slabs into one call. "
                "steel_along_span and steel_across_span MUST contain the FULL cell text "
                "(e.g. 'T12 @ 100 C/C ALT BENT UP') — the post-processor extracts dia/spacing "
                "from these strings, so dropping the spacing portion silently loses data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slab_id": {"type": "string"},
                    "thickness": {"type": ["number", "null"]},
                    "type": {"type": "string"},
                    "mix": {
                        "type": "string",
                        "description": (
                            "Concrete mix grade for this slab (e.g. M25, M20). "
                            "Use the value from the MIX column if present, "
                            "or the global concrete_grades identified in think. "
                            "If not applicable → empty string."
                        ),
                    },
                    "steel_along_span": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Full raw cell text(s) for reinforcement along span direction.",
                    },
                    "steel_across_span": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Full raw cell text(s) for reinforcement across span direction.",
                    },
                    "remarks": {"type": ["string", "null"]},
                    "source_region_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["slab_id", "steel_along_span", "steel_across_span"],
            },
        },
    },
]


# ===============================================================
# EXTRACT WITH TOOLS  (main extraction path)
# ===============================================================

def extract_with_tools(image_path, prompt_text, max_iterations=300):
    base64_image = encode_image(image_path)
    state = ExtractionState(
        project="slab",
        image_path=image_path,
        output_key="slabs",
        duplicate_key_fields=["slab_id"],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _with_tool_protocol(prompt_text)},
                _image_content(base64_image),
            ],
        }
    ]
    collected_slabs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    api_call_count = 0

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=SLAB_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        api_call_count += 1
        usage = response.usage
        if usage:
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            print(
                f"  [tokens] call #{api_call_count}: "
                f"prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  "
                f"total={usage.total_tokens}"
            )

        msg = response.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break

        tool_results = []
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if fn_name == "think":
                state.handle_think(args)
                slab_ids = args.get("slab_ids") or []
                expected = state.expected_count()
                concrete_grades = args.get("concrete_grades") or []
                steel_grades = args.get("steel_grades") or []
                print("\n============================================================")
                print("  [THINK] Structured slab extraction plan accepted")
                print(f"  slabs={len(slab_ids)} expected={expected}")
                print(f"  concrete_grades={concrete_grades}")
                print(f"  steel_grades={steel_grades}")
                print("============================================================\n")
                result_content = (
                    f"Plan accepted. {len(slab_ids)} slab rows = {expected} expected add_slab calls.\n"
                    f"Concrete grades identified: {concrete_grades}. Steel grades: {steel_grades}.\n"
                    "For each add_slab call: set mix from concrete_grades if a MIX column is not present.\n"
                    "NOW start calling add_slab immediately — one call per slab row.\n"
                    f"Slab IDs in order: {', '.join(slab_ids)}.\n"
                    "Work top-to-bottom through ALL rows. Do NOT stop until every row is recorded."
                )

            elif fn_name == "zoom_region":
                region, message = state.handle_zoom(args)
                if not region:
                    result_content = message
                else:
                    print(
                        f"  zoom_region {region['region_id']} "
                        f"({region['x1']:.2f},{region['y1']:.2f})->"
                        f"({region['x2']:.2f},{region['y2']:.2f})"
                    )
                    cropped_b64 = crop_region(
                        image_path,
                        region["x1"], region["y1"],
                        region["x2"], region["y2"],
                    )
                    result_content = [
                        {
                            "type": "text",
                            "text": (
                                f"{message} Capture full slab ID, thickness, and all bar lines. "
                                "For deleted rows, confirm the slab ID and DELETED text."
                            ),
                        },
                        _image_content(cropped_b64),
                    ]

            elif fn_name == "confirm_read":
                _, result_content = state.handle_confirm_read(args)
                print(f"  confirm_read {args.get('region_id', '')}: {str(args.get('text', ''))[:80]}")

            elif fn_name == "add_slab":
                ok, message = state.can_add_record(args)
                if not ok:
                    result_content = message
                else:
                    slab = build_slab_record(args)
                    collected_slabs.append(slab)
                    state.add_record(slab, args)
                    expected = state.expected_count() or "?"
                    remaining = (state.expected_count() or 0) - len(collected_slabs)
                    result_content = (
                        f"Recorded {len(collected_slabs)}/{expected}: slab '{slab['slab_id']}'. "
                        f"{remaining} more to go — keep calling add_slab for remaining rows."
                    )
                    reinf = slab.get("reinforcement") or {}
                    print(
                        f"  [extracted] slab={slab['slab_id']} | "
                        f"thk={slab.get('thickness')} | "
                        f"mix={slab.get('mix')} | "
                        f"dia={reinf.get('dia')} spacing={reinf.get('spacing')}"
                    )
            else:
                state.warn(f"unknown tool ignored: {fn_name}")
                result_content = f"Unknown tool '{fn_name}' ignored."

            tool_results.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_content}
            )
        messages.extend(tool_results)

    state.validate(collected_slabs)
    trace_path = state.write_trace()
    print(f"  Trace saved to {trace_path}")

    total_tokens = total_prompt_tokens + total_completion_tokens
    print(
        f"\n  ── Token usage ({os.path.basename(image_path)}) ──\n"
        f"  API calls      : {api_call_count}\n"
        f"  Prompt tokens  : {total_prompt_tokens}\n"
        f"  Completion tok : {total_completion_tokens}\n"
        f"  Total tokens   : {total_tokens}\n"
        f"  Slabs found    : {len(collected_slabs)}\n"
        f"  ─────────────────────────────────────────"
    )

    if collected_slabs:
        return json.dumps({"slabs": collected_slabs})

    print("  Tool extraction returned 0 slabs -- falling back.")
    return extract_from_image(image_path, prompt_text)


# ===============================================================
# EXTRACT FROM IMAGE  (fallback / focused-pass path)
# ===============================================================

def extract_from_image(image_path, prompt_text, retries=3):
    base64_image = encode_image(image_path)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            _image_content(base64_image),
                        ],
                    }
                ],
                temperature=0,
            )
            return clean_json_string(response.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            # Surface the REAL cause (auth, model name, rate limit, image size...)
            print(
                f"  Fallback extraction failed ({attempt}/{retries}): "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < retries:
                time.sleep(5)
    raise RuntimeError(f"Extraction failed after {retries} retries: {last_error}")


# ===============================================================
# MULTI-PASS: focused extraction helpers
# ===============================================================

_MIX_PROMPT = """Look at this slab schedule image.
Find EVERY concrete grade value (M-grade: M15, M20, M25, M30, etc.) written anywhere.
A label like "M25 : Fe500" → M25 is concrete grade, Fe500 is steel grade — extract M25.
Return ONLY: {"mix": ["M25", "M20"]}
No explanation."""

_REINF_PROMPT = """Look at this slab schedule image.
Find ALL unique reinforcement bar diameters and spacings from ALL slab rows.
Return ONLY: {"dia": ["T10", "T12"], "spacing": ["100 C/C", "150 C/C"]}
No explanation."""


def extract_mix_pass(image_path):
    """Focused pass: extract all concrete mix grades from the image."""
    import re
    print("  [pass-mix] extracting concrete mix grades...")
    raw = extract_from_image(image_path, _MIX_PROMPT)
    try:
        raw_text = raw if isinstance(raw, str) else json.dumps(raw)
        if isinstance(raw, str):
            raw = json.loads(raw)
        values = raw.get("mix", [])
        scraped = re.findall(r'\bM-?\s*\d{2,3}\b', raw_text, re.IGNORECASE)
        all_values = list(values) + scraped
        result = sorted(set(re.sub(r'M[-\s]*(\d+)', r'M\1', str(v).strip().upper()) for v in all_values if str(v).strip()))
        print(f"  [pass-mix] found: {result}")
        return result
    except Exception as e:
        print(f"  [pass-mix] parse failed: {e}")
        return []


def extract_reinf_pass(image_path):
    """Focused pass: extract all reinforcement dia/spacing from the image."""
    print("  [pass-reinf] extracting reinforcement...")
    raw = extract_from_image(image_path, _REINF_PROMPT)
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        dia = sorted(set(str(v).strip() for v in raw.get("dia", []) if str(v).strip()))
        spacing = sorted(set(str(v).strip() for v in raw.get("spacing", []) if str(v).strip()))
        print(f"  [pass-reinf] dia={dia} spacing={spacing}")
        return {"dia": dia, "spacing": spacing}
    except Exception as e:
        print(f"  [pass-reinf] parse failed: {e}")
        return {"dia": [], "spacing": []}
