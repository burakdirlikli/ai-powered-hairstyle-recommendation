"""
io_utils.py
-----------
Dosya/klasör oluşturma, JSON yazma ve URL'den görsel indirme yardımcıları.
Generator bu modüle doğrudan bağımlıdır; hiçbir iş mantığı içermez.
"""

import os
import json
import uuid
import requests
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """
    Benzersiz run ID üretir.
    Format: run_YYYYMMDD_HHMMSS_<kısa uuid>
    Örnek:  run_20260502_154312_a3f1
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:4]
    return f"run_{ts}_{short_uid}"


# ---------------------------------------------------------------------------
# Klasör Yapısı
# ---------------------------------------------------------------------------

def create_run_dir(output_root: str, run_id: str) -> str:
    """
    outputs/genai/<run_id>/ klasörünü oluşturur ve yolunu döndürür.
    """
    run_dir = os.path.join(output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def create_hairstyle_dir(run_dir: str, index: int, hairstyle_name: str) -> str:
    """
    Run klasörü içinde tek bir hairstyle için alt klasör oluşturur.
    Format: hairstyle_<index>_<isim_slug>/
    Örnek:  hairstyle_1_buzz_cut/
    """
    slug = hairstyle_name.lower().replace(" ", "_")
    folder_name = f"hairstyle_{index}_{slug}"
    hairstyle_dir = os.path.join(run_dir, folder_name)
    os.makedirs(hairstyle_dir, exist_ok=True)
    return hairstyle_dir


# ---------------------------------------------------------------------------
# JSON Yazma
# ---------------------------------------------------------------------------

def write_json(path: str, data: dict) -> None:
    """
    Verilen sözlüğü belirtilen yola UTF-8 JSON olarak yazar.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Görsel İndirme
# ---------------------------------------------------------------------------

def download_image(url: str, save_path: str, timeout: int = 15) -> str:
    """
    Verilen URL'den görseli indirir ve save_path'e kaydeder.
    Başarılıysa save_path'i döndürür, hata olursa exception fırlatır.

    TODO (ileride): retry mekanizması, timeout ayarı config'den okunabilir.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


# ---------------------------------------------------------------------------
# Görsel İşleme (Base64)
# ---------------------------------------------------------------------------

import base64

def encode_image_to_base64(image_path: str) -> str:
    """
    Belirtilen yoldaki görseli okur ve base64 string olarak döndürür.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """ISO 8601 formatında şu anki zamanı döndürür."""
    return datetime.now().isoformat()
