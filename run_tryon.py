import os
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

# ml_service dizinini path'e ekle ve VirtualTryOnManager'ı içe aktar
sys.path.append(os.path.join(current_dir, 'ml_service'))
from virtual_try_on.manager import VirtualTryOnManager

def main():
    parser = argparse.ArgumentParser(description="Oluşturulmuş bir oturum (session) klasöründeki önerilerle Virtual Try-On yapar.")
    parser.add_argument('--session', type=str, required=True, help="Oturum klasörünün yolu (Örn: outputs/session_20231024_153000)")
    args = parser.parse_args()

    session_dir = args.session
    if not os.path.isdir(session_dir):
        print(f"HATA: Belirtilen oturum klasörü bulunamadı: {session_dir}")
        return

    user_img_path = os.path.join(session_dir, "input.png")
    recommendations_dir = os.path.join(session_dir, "recommendations")

    if not os.path.exists(user_img_path):
        print(f"HATA: Oturum klasöründe girdi fotoğrafı bulunamadı: {user_img_path}")
        return

    if not os.path.isdir(recommendations_dir) or len(os.listdir(recommendations_dir)) == 0:
        print(f"HATA: Önerilen saç modelleri klasörü boş veya yok: {recommendations_dir}")
        return

    if not os.getenv("MODAL_ENDPOINT_URL"):
        print("\n[HATA] MODAL_ENDPOINT_URL .env dosyasında bulunamadı. Lütfen 'modal deploy modal_app.py' komutunu çalıştırıp URL'i .env dosyasına ekleyin.")
        return

    print("\n=== VIRTUAL TRY-ON (MODAL.COM GPU SERVISİ) ===")
    print(f"Aktif Oturum: {session_dir}")
    print(f"Bulunan Öneriler: {len(os.listdir(recommendations_dir))} adet")

    while True:
        choice = input("\nÖnerilen saç modellerinden hangisini denemek istersiniz? (1-5 arası bir sayı yazın, çıkmak için Enter): ").strip()
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
                    print(f"\n🎉 Virtual Try-On başarıyla tamamlandı! Çıktıyı '{os.path.join(session_dir, 'results')}' klasöründen inceleyebilirsiniz.")
                except Exception as e:
                    print(f"\n❌ Virtual Try-On aşamasında hata: {e}")
            else:
                print(f"\nHATA: {selected_hs_path} dosyası bulunamadı.")
        else:
            print("Lütfen geçerli bir sayı girin veya çıkmak için Enter'a basın.")

if __name__ == "__main__":
    main()
