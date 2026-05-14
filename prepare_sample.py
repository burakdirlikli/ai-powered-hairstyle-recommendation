import os
import csv
import requests
import argparse
import sys
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

# Samples klasörü her zaman proje dizininde
current_dir = os.path.dirname(os.path.abspath(__file__))
SAMPLES_BASE_DIR = os.path.join(current_dir, 'samples')

# Motoru (pipeline) içe aktar
from engine_runner import AIEnginePipeline

# ml_service dizinini path'e ekle ve VirtualTryOnManager'ı içe aktar
sys.path.append(os.path.join(current_dir, 'ml_service'))
from virtual_try_on.manager import VirtualTryOnManager

def download_image_as_png(url, save_path):
    """Verilen URL'den görseli indirir ve her zaman PNG olarak kaydeder."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path, format="PNG")
            return True
        return False
    except Exception as e:
        print(f"\nİndirme hatası: {e}")
        return False

def main():
    import datetime
    parser = argparse.ArgumentParser(description="Kullanıcı fotoğrafını alıp benzersiz bir oturum klasörü oluşturur ve tavsiye/Try-On motorunu çalıştırır.")
    parser.add_argument('--image', type=str, required=True, help="Kullanıcının girdi fotoğrafının yolu (Örn: resimler/ben.jpg)")
    args = parser.parse_args()

    input_image_path = args.image
    if not os.path.exists(input_image_path):
        print(f"HATA: Girdi fotoğrafı bulunamadı: {input_image_path}")
        return

    # Benzersiz oturum klasörü oluştur: outputs/session_YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder_name = f"session_{timestamp}"
    outputs_base_dir = os.path.join(current_dir, "outputs")
    sample_dir_path = os.path.join(outputs_base_dir, session_folder_name)
    
    os.makedirs(sample_dir_path, exist_ok=True)
    print(f"\n📁 Yeni Oturum Klasörü Oluşturuldu: {sample_dir_path}")

    # Girdi fotoğrafını oturum klasörüne 'input.png' olarak standartlaştırıp kaydet
    try:
        img = Image.open(input_image_path)
        # RGB'ye çevirerek alfa kanalı sorunlarını önle
        if img.mode != "RGB":
            img = img.convert("RGB")
        user_img_path = os.path.join(sample_dir_path, "input.png")
        img.save(user_img_path, format="PNG")
        print(f"✅ Girdi fotoğrafı kopyalandı: {os.path.join(session_folder_name, 'input.png')}")
    except Exception as e:
        print(f"HATA: Görüntü okunamadı veya kopyalanamadı: {e}")
        return
    
    # 1. Öneri Motorunu Çalıştır
    print("\n=== 1. AŞAMA: ÖNERİ MOTORU ÇALIŞTIRILIYOR ===")
    pipeline = AIEnginePipeline()
    
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
    recommendations_dir = os.path.join(sample_dir_path, "recommendations")
    os.makedirs(recommendations_dir, exist_ok=True)

    csv_path = os.path.join(current_dir, 'data', 'hs_urls.csv')
    
    if not os.path.exists(csv_path):
        print(f"HATA: {csv_path} dosyası bulunamadı!")
        return

    url_map = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url_map[row['hs_id']] = row['front_url']
            
    for idx, hs_id in enumerate(top_5_ids):
        if hs_id in url_map:
            url = url_map[hs_id]
            save_filename = f"hairstyle_{idx+1}.png"
            save_path = os.path.join(recommendations_dir, save_filename)
            
            print(f"[{idx+1}/5] İndiriliyor: {hs_id} -> {os.path.join('recommendations', save_filename)} ...", end=" ")
            success = download_image_as_png(url, save_path)
            if success:
                print("BAŞARILI")
            else:
                print("HATA!")
        else:
            print(f"UYARI: {hs_id} modeli için CSV'de URL bulunamadı.")
            
    print(f"\n=== İŞLEM TAMAMLANDI ===")
    print(f"İndirilen tüm öneri görselleri '{recommendations_dir}' klasörüne kaydedildi.")

    # 3. İsteğe Bağlı / Etkileşimli Virtual Try-On Seçimi
    if os.getenv("MODAL_ENDPOINT_URL"):
        print("\n=== 3. AŞAMA: VIRTUAL TRY-ON (MODAL.COM GPU SERVISİ) ===")
        while True:
            choice = input("\nÖnerilen saç modellerinden hangisini denemek istersiniz? (1-5 arası bir sayı yazın, çıkmak/atlamak için Enter): ").strip()
            if not choice:
                print("\nVirtual Try-On oturumu sonlandırıldı.")
                break
                
            if choice.isdigit() and 1 <= int(choice) <= 5:
                selected_num = int(choice)
                selected_hs_path = os.path.join(recommendations_dir, f"hairstyle_{selected_num}.png")
                
                if os.path.exists(selected_hs_path):
                    print(f"\nSeçilen model (hairstyle_{selected_num}.png) bulut GPU'ya aktarılıyor. Bu işlem 10-20 saniye sürebilir...")
                    try:
                        try_on_manager = VirtualTryOnManager()
                        try_on_manager.run_single_try_on(user_img_path, selected_hs_path, selected_num)
                        print("\n🎉 Virtual Try-On başarıyla tamamlandı! Çıktıyı 'results' klasöründen inceleyebilirsiniz.")
                    except Exception as e:
                        print(f"\n❌ Virtual Try-On aşamasında hata: {e}")
                else:
                    print(f"\nHATA: {selected_hs_path} dosyası bulunamadı.")
            else:
                print("Lütfen 1 ile 5 arasında geçerli bir sayı girin veya çıkmak için Enter'a basın.")
    else:
        print("\n[Bilgi] MODAL_ENDPOINT_URL .env dosyasında bulunamadı. Virtual Try-On aşaması atlandı.")

if __name__ == "__main__":
    main()
