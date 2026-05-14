"""
virtual_try_on/manager.py
--------------------------
Modal.com üzerindeki HairFastGAN GPU servisini çağırır.

Akış:
  1. Seçilen tek bir hairstyle ve user görselini base64'e encode eder
  2. Modal endpoint'ine POST atar
  3. Dönen sonucu sampleX/results/ içerisinde saç modelinin numarasıyla kaydeder
"""

import os
import base64
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class VirtualTryOnManager:
    """
    Modal.com'daki HairFastGAN servisine istek atar ve
    sonuç görselini results/ klasörüne kaydeder.
    """

    def __init__(self):
        # .env dosyasından Modal endpoint URL'ini oku
        self.endpoint_url = os.getenv("MODAL_ENDPOINT_URL")

    def run_single_try_on(self, user_image_path: str, hairstyle_image_path: str, selected_num: int):
        """
        user_image_path : Kullanıcının selfie'sinin yerel yolu (sampleX/user.png)
        hairstyle_image_path : Seçilen saç görselinin yolu (sampleX/recommendations/hairstyle_X.png)
        selected_num : Saç modelinin numarası (1-5)

        Çıktıyı sampleX/results/result_{selected_num}.png olarak kaydeder.
        """
        if not self.endpoint_url:
            raise EnvironmentError(
                "MODAL_ENDPOINT_URL .env dosyasında tanımlı değil. "
                "modal deploy sonrası oluşan URL'i ekleyin."
            )

        sample_dir = os.path.dirname(user_image_path)
        results_dir = os.path.join(sample_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        if not os.path.exists(hairstyle_image_path):
            raise FileNotFoundError(f"{hairstyle_image_path} bulunamadı.")

        def image_to_b64(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "user_image": image_to_b64(user_image_path),
            # Servis liste beklediği için tek elemanlı liste gönderiyoruz
            "hairstyle_images": [image_to_b64(hairstyle_image_path)],
        }

        response = requests.post(self.endpoint_url, json=payload, timeout=900)
        response.raise_for_status()
        data = response.json()

        result_b64 = data["results"][0]
        error = data["errors"][0]

        if error:
            print(f"  ❌ Try-On Hatası: {error}")
            return None

        result_img = Image.open(BytesIO(base64.b64decode(result_b64)))
        save_path = os.path.join(results_dir, f"result_{selected_num}.png")
        result_img.save(save_path, format="PNG")
        print(f"  ✅ Çıktı başarıyla kaydedildi: {os.path.join('results', f'result_{selected_num}.png')}")
        return save_path
