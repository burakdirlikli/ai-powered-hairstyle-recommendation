"""
virtual_try_on/manager.py
--------------------------
Yöntemler arası koordinasyon ve karşılaştırma katmanı.

Bu dosya ileride:
  - genai/ ve hf/ yöntemlerini aynı input üzerinde çalıştıracak
  - sonuçları karşılaştırıp raporlayacak
  - MLflow entegrasyonunun bağlandığı üst katman olacak

Şu an aktif iş yapan bir implementasyon içermiyor.
"""

# TODO (ileride): Karşılaştırmalı run için aktif implementasyon eklenecek
# from .genai import GenAITryOnGenerator, TryOnRequest
# from .hf import HFTryOnGenerator


class VirtualTryOnManager:
    """
    Genai ve HF yöntemlerini koordine eden üst katman.
    Şu an sadece iskelet.
    """

    def __init__(self):
        # TODO: Her iki generator da burada başlatılacak
        pass

    def run_comparison(self, user_image_path: str, recommended_hs_ids: list):
        """
        İki yöntemi aynı input üzerinde çalıştırır ve karşılaştırır.

        TODO (ileride):
          - TryOnRequest oluştur
          - genai ve hf generatorlarını çalıştır
          - sonuçları kıyasla, raporlaştır
          - MLflow'a log at
        """
        raise NotImplementedError(
            "Karşılaştırmalı run henüz implement edilmedi. "
            "Önce genai/ tarafı tamamlanacak."
        )
