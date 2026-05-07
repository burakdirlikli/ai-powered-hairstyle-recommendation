"""
config.py
---------
Genai try-on modülü için sabit ayarlar ve default değerler.
Gerçek implementasyonda bu değerler dışarıdan override edilebilir.
"""

import os

# ---------------------------------------------------------------------------
# Provider & Model
# ---------------------------------------------------------------------------

PROVIDER: str = "openai"
ANALYSIS_MODEL_NAME: str = "gpt-4o"      # Görsel analizi ve prompt üretimi için
GENERATION_MODEL_NAME: str = "dall-e-3"  # Nihai görsel üretimi için

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# virtual_try_on/outputs/genai/ olacak şekilde hesaplanır
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# _MODULE_DIR is ml_service/virtual_try_on/genai
# We want: ml_service/virtual_try_on/outputs/genai
_TRYON_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))
OUTPUT_ROOT: str = os.path.join(_TRYON_ROOT, "outputs", "genai")

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

MAX_OUTPUT_TOKENS: int = 1024  # Görsel üretim için token sınırı (gerekirse)
IMAGE_SIZE: str = "1024x1024"   # Üretilecek görsel boyutu (ileriki DALL-E veya benzeri için)
