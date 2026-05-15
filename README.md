<div align="center">
  🌐 <b>Languages:</b> <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</div>

<br>

<a id="english"></a>
# AI-Powered Hairstyle Recommendation & Virtual Try-On Pipeline (English)

> 🚀 **Production-Ready Deployment**
> This repository houses a fully complete, modular, and production-grade pipeline integrating advanced local machine learning recommendations with state-of-the-art serverless GPU Virtual Try-On (**HairFastGAN**) hosted on **Modal.com**.

---

## 📖 Architecture Overview

The system operates under a decoupled, two-tier architecture optimized for cost-efficiency, low latency, and robust failure isolation:

```
+-------------------------------------------------------------------------+
|                        LOCAL RECOMMENDATION ENGINE                      |
|                                                                         |
|  [Input Photo] ---> Face Shape & Hair Type Classifier                   |
|                              |                                          |
|                              v                                          |
|                    Candidate Selection (CSV Catalog)                    |
|                              |                                          |
|                              v                                          |
|                OpenAI GPT-4o Vision Scoring & Filtering                 |
|                              |                                          |
|                              v                                          |
|             Outputs: Top-5 Hairstyle Reference IDs & URLs               |
+-------------------------------------------------------------------------+
                                   |
            Dynamic Session Routing & Base64 Payload Encapsulation
                                   |
                                   v
+-------------------------------------------------------------------------+
|                  REMOTE CLOUD GPU BACKEND (MODAL.COM)                   |
|                                                                         |
|  POST https://...--swap-hairstyle.modal.run                             |
|  --> Warm Container Cache (@modal.enter)                                |
|  --> Persistent Volume Storage (/weights pre-loaded)                    |
|  --> Multi-Model Neural Rendering (e4e, StyleGAN2, StarGANv2, SEAN)     |
|  --> Output Swapped Image directly to Local Output Directory            |
+-------------------------------------------------------------------------+
```

### 1. Local Client-Side Engine
- **Face & Hair Attribute Classification:** Automatically predicts geometric face shape (oval, square, round, rectangular) and core hair attributes from input selfies.
- **Context-Aware Catalog Filtering:** Dynamically queries the local database (`data/hs_urls.csv`) to isolate compatible hairstyle archetypes.
- **Generative AI Scoring:** Harnesses OpenAI's multimodal vision engine to cross-reference candidates against specified personal styles (maintenance tolerance, beard synergy, lifestyle usage).

### 2. Serverless GPU Backend (Modal.com)
- **HairFastGAN Virtual Try-On Engine:** Operates deep neural rendering backbones optimized for real-time photo-realistic hairstyle transfers.
- **Build-Time Caching Optimization:** Critical deep learning checkpoints, including **CLIP (ViT-B/32)**, are securely downloaded and embedded inside the Docker container layers during the continuous integration compilation phase. This completely insulates runtime worker containers from unexpected network dropouts or `ConnectionResetError` exceptions.
- **Persistent Weight Storage:** Integrates optimized persistent network storage volumes (`/weights`) populated with structural directory hierarchies to prevent broken path traversals during cold starts.
- **Dependency Hardening:** Completely containerized over PyTorch Devel base images (`cuda11.8-cudnn8-devel`) with precise package pins (`numpy<2`, `gdown==5.1.0`) and core native build utilities (`cmake`, `dlib`) to guarantee seamless native C++ extension parsing.

---

## ✨ Core Features

- **Zero-Config Dynamic Sessions:** Simply provide any input image path via the command line. The system automatically structures a fully isolated and timestamped workspace under `outputs/session_YYYYMMDD_HHMMSS/` containing sanitized copies, candidate references, and finalized output renderings.
- **Interactive Try-On Sessions:** Run consecutive, non-blocking evaluations. Try out multiple suggested hairstyles sequentially without terminating or restarting the inference pipeline.
- **Resilient Error Isolation:** Individual model generation failures are safely intercepted and captured as granular application metrics without crashing the core client flow.

---

## 🗂️ Project Structure

```text
recommendation-system/
├── outputs/                      # Automated timestamped session history
│   └── session_YYYYMMDD_HHMMSS/
│       ├── input.png             # Standardized user selfie copy
│       ├── recommendations/      # Downloaded Top-5 hairstyle catalog visuals
│       └── results/              # Final Try-On visual compositions
├── ml_service/
│   ├── virtual_try_on/
│   │   └── manager.py            # API client routing base64 images to Modal
│   └── ...                       # Business logic & local classifiers
├── data/
│   └── hs_urls.csv               # Ground-truth catalog linking hairstyles to reference URLs
├── modal_app.py                  # Serverless Modal app configuration & cloud container definitions
├── run_recommendation.py         # Step 1: AI recommendation script to build session folder
├── run_tryon.py                  # Step 2: Interactive Try-On script executing Modal inference
├── engine_runner.py              # Primary interface powering backend candidate generation
└── requirements.txt              # Local client dependencies
```

---

## 🛠️ Local Setup & Installation

### Prerequisites
- **Python:** Version 3.10+ recommended.
- **Git:** Make sure Git and Git-LFS are installed for repository syncing.

### Step-by-Step Installation

1. **Clone the Project:**
   ```bash
   git clone <repository-url>
   cd recommendation-system
   ```

2. **Initialize an Isolated Environment:**
   ```bash
   python -m venv venv
   ```
   - **Windows:**
     ```cmd
     venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

3. **Install Client Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ☁️ Deploying the Serverless GPU Backend

Before executing client-side tasks, ensure the dedicated serverless backend app is running securely in your cloud workspace.

1. **Authenticate Modal CLI:**
   ```bash
   modal setup
   ```

2. **Deploy the Inference App:**
   ```bash
   modal deploy modal_app.py
   ```
   > **Note:** Upon successful initialization, the console outputs a persistent cloud service endpoint mapping directly to your authenticated environment.

---

## 🚀 Execution Guide

With your backend deployed, run full client inference using the newly implemented dynamic session engine.

The workflow is divided into two highly modular scripts.

### Step 1: Analysis & Recommendation
Pass the target user picture directly to the recommendation engine:

```bash
python run_recommendation.py --image "test-user.png"
```
> **Note:** To test with your own photo, provide the full absolute or relative path to the image (e.g., `--image "C:\Users\username\Desktop\my_photo.jpg"`).

- **Workspace Generation:** Automatically instantiates a brand new target folder under `outputs/session_YYYYMMDD_HHMMSS/`.
- **Candidate Downloads:** Evaluates face structures, ranks candidates, and securely populates `recommendations/` with reference images.

### Step 2: Virtual Try-On Execution
Using the generated session folder, initiate the interactive Try-On interface:

```bash
python run_tryon.py --session "outputs/session_YYYYMMDD_HHMMSS"
```
- **Interactive Multi-Pass Try-On:** The application triggers an ongoing loop to test multiple top hairstyles interactively:
   ```text
   Önerilen saç modellerinden hangisini denemek istersiniz? (1-5 arası bir sayı yazın, çıkmak için Enter):
   ```
- Enter `1`, `2`, `3`, etc., to pipe specific source configurations directly into cloud serverless instances.
- Finished swap images are persisted seamlessly to your local `results/` folder inside the session directory.

---

## 📜 License

Developed as a highly decoupled production-grade visual evaluation module. All internal business implementations and related media artifacts are reserved.

<br>
<hr>
<br>

<a id="türkçe"></a>
# Yapay Zeka Destekli Saç Modeli Öneri ve Sanal Deneme (Try-On) Hattı (Türkçe)

> 🚀 **Üretime Hazır Mimari (Production-Ready)**
> Bu depo, lokal makine öğrenimi tabanlı gelişmiş tavsiye motoru ile **Modal.com** üzerindeki yüksek performanslı sunucusuz GPU'larda koşan **HairFastGAN** sanal saç transferi (Virtual Try-On) altyapısını baştan sona entegre eden modüler bir projedir.

---

## 📖 Mimariye Genel Bakış

Sistem, maliyet tasarrufu, düşük gecikme süresi ve hata izolasyonu sağlamak amacıyla iki katmanlı (decoupled) bir altyapıda çalışır:

```
+-------------------------------------------------------------------------+
|                         LOKAL ÖNERİ MOTORU KATMANI                      |
|                                                                         |
|  [Girdi Selfie] ---> Yüz Şekli ve Saç Tipi Sınıflandırıcısı             |
|                              |                                          |
|                              v                                          |
|                   Aday Filtreleme (CSV Veritabanı)                      |
|                              |                                          |
|                              v                                          |
|               OpenAI GPT-4o Vision ile Skorlama ve Seçim                |
|                              |                                          |
|                              v                                          |
|             Çıktı: En İyi 5 Saç Modeli ID'si ve Görsel URL'i            |
+-------------------------------------------------------------------------+
                                   |
             Dinamik Oturum Yönetimi & Base64 Paketleme Akışı
                                   |
                                   v
+-------------------------------------------------------------------------+
|                  BULUT GPU ARKA PLANI (MODAL.COM)                       |
|                                                                         |
|  POST https://...--swap-hairstyle.modal.run                             |
|  --> Önceden Isıtılmış Bellek Önbelleği (@modal.enter)                  |
|  --> Kalıcı Disk Birimi (Önceden yüklenmiş /weights klasörü)            |
|  --> Çoklu Yapay Zeka Sentezi (e4e, StyleGAN2, StarGANv2, SEAN)         |
|  --> Transferi Tamamlanan Görseli Doğrudan Lokal Dizine Yaz             |
+-------------------------------------------------------------------------+
```

### 1. Lokal İstemci Katmanı
- **Yüz ve Saç Analizi:** Kullanıcının yüklediği fotoğraftan geometrik yüz şeklini (oval, kare, yuvarlak, dikdörtgen) ve mevcut saç yapısını otomatik olarak tahmin eder.
- **Katalog Filtreleme:** Lokal veritabanını (`data/hs_urls.csv`) tarayarak analiz sonuçlarıyla eşleşen aday saç modellerini listeler.
- **Üretken Yapay Zeka Puanlaması:** OpenAI'ın çoklu modlu (multimodal) görüntü işleme motorunu kullanarak adayları kullanıcının kişisel tercihleriyle (bakım zorluğu, sakal durumu, kullanım tarzı) çapraz sorgular ve en iyi 5 modeli belirler.

### 2. Sunucusuz GPU Arka Planı (Modal.com)
- **HairFastGAN Motoru:** Gerçek zamanlı ve fotoğraf gerçekliğinde saç transferleri yapabilen gelişmiş derin sinir ağı omurgalarını çalıştırır.
- **Derleme Zamanı Önbellekleme (Build-Time Caching):** Container ayağa kalkarken yaşanabilecek anlık ağ kopmalarını veya `ConnectionResetError` hatalarını tamamen önlemek için, **CLIP (ViT-B/32)** gibi kritik model ağırlıkları imajın bulutta derlenme aşamasında (`run_commands`) indirilip kalıcı katmanlara gömülür.
- **Kalıcı Ağırlık Depolaması:** İlk kurulumda kırık sembolik link (broken symlink) hatalarının önüne geçmek için optimize edilmiş bulut depolama birimleri (`/weights`) ve alt klasör hiyerarşileri kullanılır.
- **Zırhlı Ortam Bağımlılıkları:** PyTorch Devel taban imajı (`cuda11.8-cudnn8-devel`) üzerinde `numpy<2` ve `gdown==5.1.0` gibi kararlı sürümler sabitlenmiş; yerel C++ uzantılarının sorunsuz derlenmesi için `cmake` ve `dlib` sistem kütüphaneleriyle güçlendirilmiştir.

---

## ✨ Temel Özellikler

- **Sıfır Ayar Dinamik Oturumlar:** Terminalden herhangi bir resmin yolunu vermeniz yeterlidir. Sistem otomatik olarak `outputs/session_YYYYMMDD_HHMMSS/` formatında zaman damgalı izole bir oturum klasörü açar; orijinal resmi standartlaştırır, önerileri indirir ve nihai çıktıları tek bir yerde toplar.
- **Etkileşimli Deneme Akışı:** Arka arkaya kesintisiz denemeler yapabilirsiniz. Betiği kapatıp açmaya gerek kalmadan motorun önerdiği saç modellerini sırayla konsol üzerinden seçip test edebilirsiniz.
- **Hata İzolasyonu:** Modellerden birinde oluşabilecek herhangi bir matris veya hizalama hatası ana akışı çökertmez; loglanarak atlanır ve diğer modellerin denenmesine olanak tanır.

---

## 🗂️ Proje Yapısı

```text
recommendation-system/
├── outputs/                      # Otomatik zaman damgalı oturum geçmişi
│   └── session_YYYYMMDD_HHMMSS/
│       ├── input.png             # Standartlaştırılmış kullanıcı fotoğrafı
│       ├── recommendations/      # İndirilen en iyi 5 referans saç görseli
│       └── results/              # Üretilen nihai Try-On sonuçları
├── ml_service/
│   ├── virtual_try_on/
│   │   └── manager.py            # Base64 görsellerini Modal.com'a ileten istemci
│   └── ...                       # İş mantığı ve sınıflandırma fonksiyonları
├── data/
│   └── hs_urls.csv               # Saç modellerini görsel URL'leriyle eşleyen veritabanı
├── modal_app.py                  # Sunucusuz bulut GPU servisi ve imaj tanımları
├── run_recommendation.py         # Adım 1: Oturum klasörü oluşturan AI analiz/öneri betiği
├── run_tryon.py                  # Adım 2: Modal bulutunda deneme işlemini yapan etkileşimli betik
├── engine_runner.py              # Aday önerme algoritmalarını yöneten ana sınıf
└── requirements.txt              # Lokal Python bağımlılıkları
```

---

## 🛠️ Lokal Kurulum

### Ön Koşullar
- **Python:** 3.10 veya üzeri tavsiye edilir.
- **Git:** Büyük dosyaların (model ağırlıkları vb.) çekilebilmesi için Git ve Git-LFS kurulu olmalıdır.

### Adım Adım Kurulum

1. **Projeyi Klonlayın:**
   ```bash
   git clone <repository-url>
   cd recommendation-system
   ```

2. **Sanal Ortam Oluşturun ve Aktifleştirin:**
   ```bash
   python -m venv venv
   ```
   - **Windows:**
     ```cmd
     venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

3. **Gerekli Paketleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ☁️ Bulut GPU Arka Planını Dağıtma (Deploy)

Lokal betikleri çalıştırmadan önce, saç transferini yapacak olan bulut servisinin Modal.com üzerinde aktif olduğundan emin olun.

1. **Modal CLI Kimlik Doğrulaması:**
   ```bash
   modal setup
   ```

2. **Servisi Buluta Gönderin:**
   ```bash
   modal deploy modal_app.py
   ```
   > **Not:** Kurulum başarıyla tamamlandığında konsol size doğrudan bulut ortamınızla eşleşen kalıcı bir servis URL'i çıktısı verecektir.

---

## 🚀 Kullanım Rehberi

Arka plan servisiniz bulutta hazır olduktan sonra, yeni dinamik oturum motorunu kullanarak analiz ve deneme işlemlerini başlatabilirsiniz.

İş akışı daha modüler bir yapı sağlamak için iki ayrı betiğe ayrılmıştır.

### Adım 1: Analiz ve Öneri
Kullanıcının fotoğrafını doğrudan öneri motoruna parametre olarak verin:

```bash
python run_recommendation.py --image "test-user.png"
```
> **Not:** Kendi fotoğrafınızla denemek isterseniz, görselin bilgisayarınızdaki tam yolunu (tam dosya dizinini) veya göreceli yolunu vermeniz gerekmektedir (Örn: `--image "C:\Kullanicilar\isim\Masaustu\fotograf.jpg"`).

- **Oturum Açma:** Betik `outputs/session_YYYYMMDD_HHMMSS/` adında yepyeni bir dizin hazırlar.
- **Aday İndirme:** Yüz hatları analiz edilir, OpenAI destekli motor en uygun 5 modeli seçer ve referans görsellerini `recommendations/` klasörüne indirir. İşlem sonunda oluşturulan oturum dizininin adını verir.

### Adım 2: Sanal Deneme (Virtual Try-On)
Oluşturulan oturum klasörünü kullanarak etkileşimli deneme akışını başlatın:

```bash
python run_tryon.py --session "outputs/session_YYYYMMDD_HHMMSS"
```
- **Konsol Üzerinden Canlı Deneme:** Uygulama size konsolda şu soruyu yöneltir ve bir döngüye girer:
   ```text
   Önerilen saç modellerinden hangisini denemek istersiniz? (1-5 arası bir sayı yazın, çıkmak için Enter):
   ```
- `1`, `2`, `3` gibi sayılar girerek dilediğiniz saçı anında bulut sunucusunda kendi resminize uygulayabilirsiniz.
- Tamamlanan resimler otomatik olarak ilgili oturumun içerisindeki lokal `results/` klasörüne kaydedilir.

---

## 📜 Lisans

Gelişmiş ve izole bir yapay zeka modülü olarak tasarlanmıştır. Tüm hakları saklıdır.