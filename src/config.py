import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL        = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "high")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")
