"""
vision_extractor.py - Slab project

Tool-augmented extraction with enforced sequence:
think -> zoom_region -> confirm_read -> add_slab.
"""

import base64
import io
import json
import time

from PIL import Image
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_IMAGE_DETAIL, OPENAI_MODEL
from extraction_guard import ExtractionState, build_slab_record, clean_json_string


client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def _image_content(base64_image, detail=OPENAI_IMAGE_DETAIL):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": detail,
        },
    }


def _crop_image_b64(image_path, x1, y1, x2, y2):
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
    left = int(x1 * w)
    top = int(y1 * h)
    right = max(int(x2 * w), left + 20)
    bottom = max(int(y2 * h), top + 20)
    cropped = img.crop((left, top, right, bottom))
    longest = max(cropped.size)
    if longest < 1200:
        scale = 1200 / longest
        cropped = cropped.resize(
            (int(cropped.width * scale), int(cropped.height * scale)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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
        "You MUST use tools — do NOT return raw JSON text.\n"
        "Step 1: call think() with your full extraction plan.\n"
        "Step 2: call add_slab() once for EVERY slab row (S1, S2, ... top to bottom).\n"
        "  - Do NOT stop early. Every visible slab row must get its own add_slab call.\n"
        "Optional: call zoom_region() + confirm_read() for any cell that is blurry or ambiguous.\n\n"
        f"{prompt_text}"
    )


SLAB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": (
                "Mandatory first call. Return structured observable slab schedule planning data, "
                "including table columns, slab IDs, deleted rows, expected count, and zoom targets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_quality": {"type": "string"},
                    "table_bounds": {"type": "object"},
                    "table_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Visible schedule columns, such as TYPE, THICKNESS, ALONG SPAN, ACROSS SPAN.",
                    },
                    "slab_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All slab IDs visible in the TYPE column.",
                    },
                    "deleted_rows": {"type": "array", "items": {"type": "string"}},
                    "multi_line_cells": {"type": "array", "items": {"type": "string"}},
                    "expected_count": {"type": "integer"},
                    "zoom_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "region_id": {"type": "string"},
                                "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                                "target": {"type": "string"},
                                "x1": {"type": "number"},
                                "y1": {"type": "number"},
                                "x2": {"type": "number"},
                                "y2": {"type": "number"},
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
            "description": "Crop and zoom a planned slab schedule region after think.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "purpose": {"type": "string", "enum": _REGION_PURPOSE_ENUM},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                    "x2": {"type": "number"},
                    "y2": {"type": "number"},
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
                "Record ONE slab row. Call this once per slab ID (S1, S2, ...). "
                "Do not batch multiple slabs into one call. "
                "steel_along_span and steel_across_span MUST contain the FULL "
                "cell text for that slab (e.g. 'T12 @ 100 C/C ALT BENT UP'), "
                "not just the diameter token. The post-processor extracts "
                "diameters and spacings from these strings, so dropping the "
                "spacing portion silently loses data. "
                "zoom_region/confirm_read are optional — use only for blurry or ambiguous cells."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slab_id": {"type": "string"},
                    "thickness": {"type": ["number", "null"]},
                    "type": {"type": "string"},
                    "mix": {"type": "string"},
                    "steel_along_span": {"type": "array", "items": {"type": "string"}},
                    "steel_across_span": {"type": "array", "items": {"type": "string"}},
                    "remarks": {"type": ["string", "null"]},
                    "source_region_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "slab_id",
                    "steel_along_span",
                    "steel_across_span",
                ],
            },
        },
    },
]


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

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=SLAB_TOOLS,
            tool_choice="auto",
            temperature=0,
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
                print("\n============================================================")
                print("  [THINK] Structured slab extraction plan accepted")
                print(f"  slabs={len(slab_ids)} expected={expected}")
                print("============================================================\n")
                result_content = (
                    f"Plan accepted. {len(slab_ids)} slab rows identified = {expected} expected add_slab calls.\n"
                    "NOW start calling add_slab immediately — one call per slab row.\n"
                    f"Slab IDs in order: {', '.join(slab_ids)}.\n"
                    "Work top-to-bottom through ALL rows. Do NOT stop until every row has been recorded. "
                    "Use zoom_region only if a cell is genuinely unclear."
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
                    cropped_b64 = _crop_image_b64(
                        image_path,
                        region["x1"],
                        region["y1"],
                        region["x2"],
                        region["y2"],
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
                    print(f"  add_slab: {slab['slab_id']}")
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
    if collected_slabs:
        print(f"  {len(collected_slabs)} slab(s).")
        return json.dumps({"slabs": collected_slabs})

    print("  Tool extraction returned 0 slabs -- falling back.")
    return extract_from_image(image_path, prompt_text)


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
            if attempt < retries:
                print(f"  Fallback extraction failed, retrying ({attempt}/{retries})...")
                time.sleep(5)
    raise RuntimeError(f"Extraction failed after {retries} retries: {last_error}")
