"""
schemas.py
----------
Input / output veri yapıları.
Generator ve io_utils arasında tutarlı veri akışı sağlar.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class HairstyleInput:
    """
    Bir hairstyle önerisini temsil eden standardize input yapısı.
    Her hairstyle için en az name ve image_url zorunludur.
    """
    name: str
    image_url: str
    hs_id: Optional[str] = None          # Opsiyonel: katalog ID (örn. "HS04")
    score: Optional[float] = None        # Opsiyonel: öneri skoru


@dataclass
class TryOnRequest:
    """
    Generator'a iletilen tam istek nesnesi.
    """
    user_image_path: str                 # Kullanıcı fotoğrafının lokal yolu
    hairstyles: list                     # List[HairstyleInput]
    prompt_version: str = "v1"          # Kullanılacak prompt versiyonu
    run_id: Optional[str] = None        # Dışarıdan verilebilir; yoksa otomatik üretilir


# ---------------------------------------------------------------------------
# Per-Hairstyle Output
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Tek bir hairstyle için üretim sonucu.
    Başarılı veya başarısız her hairstyle için oluşturulur.
    """
    hairstyle_name: str
    hairstyle_image_url: str
    status: str                          # "success" | "failed"

    # Yollar
    reference_image_path: Optional[str] = None
    output_image_path: Optional[str] = None

    # Prompt & model bilgileri
    prompt_version: Optional[str] = None
    prompt_text: Optional[str] = None
    provider: Optional[str] = None
    model_name: Optional[str] = None

    # Performans & izleme
    latency_sec: Optional[float] = None
    request_id: Optional[str] = None    # API'den dönen request ID (MLflow için)
    raw_response_metadata: Optional[dict] = None  # Ham API yanıtı (MLflow için)

    # Hata
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Run-Level Output
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """
    Bir run'ın tamamını özetleyen yapı.
    run_summary.json olarak kaydedilir.
    """
    run_id: str
    method: str                          # "genai"
    provider: str
    model_name: str
    prompt_version: str
    user_image_path: str
    total_hairstyles: int
    successful: int
    failed: int
    total_latency_sec: float
    output_dir: str
    results: list = field(default_factory=list)  # List[dict] — GenerationResult'ların serileşmiş hali
    timestamp: Optional[str] = None
