"""
prompt_builder.py - Slab project

Builds the extraction prompt from a shared base template + per-pattern rules file.

Layout:
    prompts/
        _base.txt              shared template with $placeholders
        rules/
            pattern_1.txt      per-pattern rules only
            ...
            pattern_9.txt
"""

import os
from string import Template


PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
RULES_DIR = os.path.join(PROMPTS_DIR, "rules")
BASE_PATH = os.path.join(PROMPTS_DIR, "_base.txt")


SLAB_SCHEMA = """{
  "slabs": [
    {
      "slab_id": "",
      "thickness": null,
      "type": "",
      "mix": "",
      "reinforcement": {
        "dia": [],
        "spacing": []
      }
    }
  ]
}"""


DEFAULT_ROLE = (
    "You are an expert structural drawing data extractor specialized in RCC slab schedules."
)


PATTERN_CONFIG = {
    1: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    2: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    3: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    4: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    5: {"role": "You are an expert structural drawing data extractor specialized in RCC slab reinforcement schedules."},
    6: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    7: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    8: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
    9: {"role": "You are an expert structural drawing data extractor specialized in RCC slab schedules."},
}


def _read(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _rules_path(pattern_number):
    return os.path.join(RULES_DIR, f"pattern_{pattern_number}.txt")


def build_prompt(pattern_number):
    pattern_number = int(pattern_number)
    config = PATTERN_CONFIG.get(pattern_number)
    if config is None:
        raise ValueError(f"Unknown slab pattern: {pattern_number}")

    rules_path = _rules_path(pattern_number)
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Missing rules file for pattern {pattern_number}: {rules_path}")

    template = Template(_read(BASE_PATH))
    return template.substitute(
        role=config.get("role", DEFAULT_ROLE),
        json_schema=SLAB_SCHEMA,
        pattern_rules=_read(rules_path).strip(),
    )


if __name__ == "__main__":
    for n in sorted(PATTERN_CONFIG):
        prompt = build_prompt(n)
        print(f"--- pattern {n}: {len(prompt)} chars ---")
