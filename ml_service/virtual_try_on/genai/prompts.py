"""
prompts.py
----------
Versiyonlanabilir prompt sistemi.

Öncelik sırası (tüm versiyonlarda korunmalı):
  1. Kimlik korunumu  — kullanıcı aynı kişi olarak kalmalı
  2. Saç modeline benzerlik — referans saç stili doğru yansıtılmalı
  3. Doğallık — sonuç mümkün olduğunca gerçekçi görünmeli
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Prompt versiyonları
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "v1": (
        "Image A is the user. Preserve Image A as much as possible. Image B is only the hairstyle reference. Transfer only the hairstyle from Image B onto the person in Image A. Do not change anything else."
    ),
    # v2, v3... ileride buraya eklenecek
}

SUPPORTED_VERSIONS = list(_PROMPTS.keys())
DEFAULT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prompt(version: Optional[str] = None) -> str:
    """
    Verilen versiyona ait prompt metnini döndürür.
    Version verilmezse DEFAULT_VERSION kullanılır.
    Bilinmeyen version verilirse ValueError fırlatır.
    """
    v = version or DEFAULT_VERSION
    if v not in _PROMPTS:
        raise ValueError(
            f"Unknown prompt version: '{v}'. Supported: {SUPPORTED_VERSIONS}"
        )
    return _PROMPTS[v]


def build_prompt(version: Optional[str] = None, **kwargs) -> str:
    """
    İleride dinamik parametreler (hairstyle_name vb.) ile
    prompt'u özelleştirmek için genişletilebilir builder.
    Şu an sadece temel prompt metnini döndürür.

    TODO (v2+): kwargs ile hairstyle_name, user_description gibi
    bilgileri prompt template'e enjekte et.
    """
    return get_prompt(version)
