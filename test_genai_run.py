import os
import sys
import json
from dotenv import load_dotenv

# Path ayarı: ml_service'i görebilmesi için
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'ml_service'))

from virtual_try_on.genai import GenAITryOnGenerator, TryOnRequest, HairstyleInput

def run_test():
    load_dotenv()
    
    print("--- GenAI Try-On Uçtan Uca Test Başlatılıyor ---")
    
    # 1. İstek parametrelerini hazırla
    # Not: sample-1.jpg'nin varlığından eminiz (önceki adımlarda oluşturduk)
    user_img = os.path.join(current_dir, "samples", "sample-1.jpg")
    
    hairstyles = [
        HairstyleInput(
            name="Buzz Cut", 
            image_url="https://raw.githubusercontent.com/burakdirlikli/hairstyle-catalog-images/main/images/HS04_front.jpg"
        ),
        HairstyleInput(
            name="Middle Part", 
            image_url="https://raw.githubusercontent.com/burakdirlikli/hairstyle-catalog-images/main/images/HS14_front.jpg"
        )
    ]
    
    request = TryOnRequest(
        user_image_path=user_img,
        hairstyles=hairstyles,
        prompt_version="v1"
    )
    
    # 2. Üretimi Başlat
    try:
        generator = GenAITryOnGenerator()
        summary = generator.run(request)
        
        print("\n--- Test Tamamlandı ---")
        print(f"Run ID: {summary.run_id}")
        print(f"Toplam Süre: {summary.total_latency_sec} sn")
        print(f"Başarılı: {summary.successful}")
        print(f"Hatalı: {summary.failed}")
        print(f"Çıktı Dizini: {summary.output_dir}")
        
        # Sonuç özetini ekrana bas
        summary_file = os.path.join(summary.output_dir, "run_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print("\nRun Summary JSON (Kısmi):")
                print(json.dumps({k: data[k] for k in ["run_id", "status_counts"] if k in data} or data, indent=2))

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test başarısız oldu: {e}")

if __name__ == "__main__":
    run_test()
