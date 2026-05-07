"""
virtual_try_on/hf/generator.py
-------------------------------
Yöntem 2: Pretrained / HuggingFace tabanlı visual try-on.

Bu dosya şu an bir placeholder iskelet.
İleride bu yöntemin implementasyonu buraya eklenecek.

Planlanan akış:
  - HuggingFace üzerinden uygun bir pretrained try-on modeli yükle
  - Kullanıcı fotoğrafı + referans saç görseli → model → çıktı görseli
  - Aynı output klasör yapısına ve metadata standardına uyacak
"""


class HFTryOnGenerator:
    """
    HuggingFace pretrained model tabanlı visual try-on motoru.

    TODO: İmplementasyon eklenecek.
    """

    def __init__(self):
        # TODO: Model yükleme burada yapılacak
        pass

    def run(self, request):
        """
        TODO: HuggingFace modeliyle üretim akışı burada implement edilecek.
        Girdi/çıktı formatı genai tarafıyla uyumlu olacak.
        """
        raise NotImplementedError(
            "HuggingFace try-on henüz implement edilmedi."
        )
