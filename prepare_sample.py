import os
import csv
import requests
import argparse
import sys

# Motoru (pipeline) içe aktar
from engine_runner import AIEnginePipeline

from PIL import Image
from io import BytesIO

def download_image_as_png(url, save_path):
    """Verilen URL'den görseli indirir ve her zaman PNG olarak kaydeder."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # RGBA (şeffaflık) varsa ve kaydederken sorun olmasın diye RGB'ye çevirebiliriz,
            # ancak PNG şeffaflığı desteklediği için direkt kaydedebiliriz.
            img.save(save_path, format="PNG")
            return True
        return False
    except Exception as e:
        print(f"\nİndirme hatası: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verilen sample klasörü için tavsiye motorunu çalıştırıp referans saç görsellerini indirir.")
    parser.add_argument('--sample', type=str, required=True, help="Örnek klasör adı (Örn: sample3)")
    args = parser.parse_args()

    sample_folder = args.sample
    sample_id = sample_folder.replace("sample", "") # "sample3" -> "3"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir_path = os.path.join(current_dir, 'samples', sample_folder)
    
    # Klasör yoksa oluştur
    if not os.path.exists(sample_dir_path):
        os.makedirs(sample_dir_path, exist_ok=True)
        print(f"Klasör oluşturuldu: {sample_dir_path}")
        
    # User görselini bul (içinde 'user' geçen ilk .png/.jpg dosyası)
    user_img_path = None
    for f in os.listdir(sample_dir_path):
        if 'user' in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            user_img_path = os.path.join(sample_dir_path, f)
            break
            
    if not user_img_path:
        print(f"HATA: '{sample_folder}' klasöründe 'user' ismini içeren bir fotoğraf bulunamadı.")
        print(f"Lütfen '{sample_dir_path}' klasörünün içine kullanıcı fotoğrafını (Örn: user.jpg) ekleyin.")
        return

    print(f"Kullanıcı fotoğrafı bulundu: {os.path.basename(user_img_path)}")
    
    # 1. Öneri Motorunu Çalıştır
    print("\n=== 1. AŞAMA: ÖNERİ MOTORU ÇALIŞTIRILIYOR ===")
    pipeline = AIEnginePipeline()
    
    # Test için varsayılan tercihler (İsterseniz parametre olarak dışarıdan da alabilirsiniz)
    prefs = {
        "hair_length": "medium",
        "maintenance": "low",
        "beard": "yes",
        "usage": "casual"
    }
    
    result = pipeline.run(user_img_path, prefs)
    if not result:
        print("\nHATA: Öneri motoru sonuç döndüremedi. Lütfen OpenAI API anahtarınızı veya yüz analizi kütüphanenizi kontrol edin.")
        return
        
    top_5_ids, _ = result
    print(f"\nMotorun Önerdiği Saç Modelleri: {top_5_ids}")
    
    # 2. URL'leri Bul ve Görselleri İndir
    print("\n=== 2. AŞAMA: REFERANS SAÇ GÖRSELLERİ İNDİRİLİYOR ===")
    csv_path = os.path.join(current_dir, 'data', 'hs_urls.csv')
    
    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} dosyası bulunamadı!")
        return

    # CSV'yi oku ve bir sözlüğe (dictionary) aktar
    url_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url_map[row['hs_id']] = row['front_url']
            
    # Görselleri sırayla indir
    for idx, hs_id in enumerate(top_5_ids):
        if hs_id in url_map:
            url = url_map[hs_id]
            # Her zaman .png olarak kaydedelim
            save_filename = f"hairstyle_{idx+1}.png"
            save_path = os.path.join(sample_dir_path, save_filename)
            
            print(f"[{idx+1}/5] İndiriliyor: {hs_id} -> {save_filename} ...", end=" ")
            success = download_image_as_png(url, save_path)
            if success:
                print("BAŞARILI")
            else:
                print("HATA!")
        else:
            print(f"UYARI: {hs_id} modeli için CSV'de URL bulunamadı.")
            
    print(f"\n=== İŞLEM TAMAMLANDI ===")
    print(f"İndirilen tüm görseller '{sample_dir_path}' klasörüne kaydedildi.")
    print("Artık bu klasörü doğrudan Colab'a yükleyebilir ve Virtual Try-On için kullanabilirsiniz.")

if __name__ == "__main__":
    main()
