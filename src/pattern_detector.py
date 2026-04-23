import os
import json

from pdf_to_images import convert_pdf_to_images
from vision_extractor import extract_from_image


def detect_pattern(pdf_path, temp_folder):
    """
    Detect slab schedule pattern (1 to 5)
    based strictly on HEADER structure.
    """

    image_paths = convert_pdf_to_images(pdf_path, temp_folder)

    if not image_paths:
        raise Exception("No image generated for pattern detection.")

    first_image = image_paths[0]

    classification_prompt = """
You are an expert at identifying RCC slab schedule header patterns.

Look ONLY at the HEADER structure.
Ignore data rows completely.

There are EXACTLY 5 slab patterns.

Return ONLY one number:
1
2
3
4
5
6
7
8
9

No explanation.
No extra text.

=========================
CRITICAL DIFFERENTIATION RULES
=========================

1. If headers contain BOTH:
   - "MAIN REINF." AND "DISTRIBUTION REINF."
   → RETURN 8

2. If headers are SIMPLE (few columns, no short/long separation)
   → DO NOT select 9

3. Pattern 9 is ONLY when:
   - BOTH MAIN + DISTRIBUTION exist
   INSIDE:
   - BOTTOM REINFORCEMENT
   - TOP SUPPORT REINFORCEMENT

   Example:
   BOTTOM REINFORCEMENT (MAIN + DISTRIBUTION)
   TOP SUPPORT REINFORCEMENT (MAIN + DISTRIBUTION)

   → ONLY THEN RETURN 9

4. Pattern 6 / 7 detection:

   If headers contain:
   - BOTTOM REINFORCEMENT ALONG SHORT SPAN
   - BOTTOM REINFORCEMENT ALONG LONG SPAN
   - TOP SUPPORT REINFORCEMENT ALONG SHORT SPAN
   - TOP SUPPORT REINFORCEMENT ALONG LONG SPAN

   AND DO NOT explicitly split into MAIN vs DISTRIBUTION

   → RETURN 6 or 7 (NOT 9)

5. Difference between 6 and 7:

   - Pattern 6 contains:
     "MAIN REINF." and "DISTRIBUTION REINF." labels in header description

   - Pattern 7 does NOT explicitly label them as MAIN/DISTRIBUTION
     but only uses SHORT/LONG span wording

=========================
PATTERN 1
=========================

Columns:
- TYPE
- THICKNESS
- STEEL ALONG SPAN
- STEEL ACROSS SPAN
- REMARKS

=========================
PATTERN 2
=========================

Columns:
- TYPE
- THICKNESS
- STEEL ALONG SHORT SPAN
- STEEL ACROSS SHORT SPAN
- REMARKS

=========================
PATTERN 3
=========================

Columns:
- SLAB MARKED
- THK.
- TYPE
- REINFORCEMENT
    - SHORT BAR
    - LONG BAR
- REMARKS

=========================
PATTERN 4
=========================

Columns:
- SLAB NO.
- SLAB THK.
- MIX
- TYPE
- MAIN STEEL
    - 11 TO SHORT SPAN
    - 11 TO LONG SPAN
- EX. TOP OVER SUPPORT
    - 11 TO SHORT SPAN
    - 11 TO LONG SPAN
- DIST. STEEL

=========================
PATTERN 5
=========================

Columns:
- BAR MARKED
- DIA
- SPACING
- REMARKS

=========================
PATTERN 6
=========================

Columns:
- SLAB MARKED
- SLAB THICKNESS
- BOTTOM REINFORCEMENT - ALONG SHORT SPAN (MAIN REINF.)
- BOTTOM REINFORCEMENT - ALONG LONG SPAN (DISTRIBUTION REINF.)
- TOP SUPPORT REINFORCEMENT - ALONG SHORT SPAN
- TOP SUPPORT REINFORCEMENT - ALONG LONG SPAN
- IN MID SPAN (TO BE LAPPED WITH SUPPORT REINF.)
- DISTRIBUTION
- REMARKS

=========================
PATTERN 7
=========================

Columns:
- SLAB MARKED
- SLAB THICKNESS
- BOTTOM REINFORCEMENT - ALONG SHORT SPAN
- BOTTOM REINFORCEMENT - ALONG LONG SPAN
- TOP SUPPORT REINFORCEMENT - ALONG SHORT SPAN
- TOP SUPPORT REINFORCEMENT - ALONG LONG SPAN
- IN MID SPAN (TO BE LAPPED WITH SUPPORT REINF.)
- DISTRIBUTION
- REMARKS

=========================
PATTERN 8
=========================

Columns:
- NOS.
- THK.
- MAIN REINF.
- DISTRIBUTION REINF.
- REMARKS

=========================
PATTERN 9
=========================

Columns:
- SLAB MARKED
- SLAB THICKNESS
- BOTTOM REINFORCEMENT - (MAIN REINF.)
- BOTTOM REINFORCEMENT - (DISTRIBUTION REINF.)
- TOP SUPPORT REINFORCEMENT - (MAIN REINF.)
- TOP SUPPORT REINFORCEMENT - (DISTRIBUTION REINF.)
- IN MID SPAN (TO BE LAPPED WITH SUPPORT REINF.)
- DISTRIBUTION
- REMARKS

=========================

Return ONLY the number.
"""

    result = extract_from_image(first_image, classification_prompt)
    result = result.strip()

    if not result.isdigit():
        raise Exception(f"Pattern detection failed. Model returned: {result}")

    return int(result)