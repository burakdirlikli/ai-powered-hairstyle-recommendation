"""
modal_app.py — HairFastGAN Virtual Try-On Servisi
Kullanım: modal deploy modal_app.py
"""

import modal
import base64
import io
from pathlib import Path

# ── Docker Image Tanımı ───────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel",
        add_python="3.10",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(["git", "git-lfs", "ninja-build", "libgl1", "libglib2.0-0", "g++", "build-essential", "cmake"])
    .pip_install([
        "numpy<2", "fastapi[standard]", "addict", "appdirs", "ftfy", "gdown==5.1.0", "face_alignment",
        "opencv-python-headless", "scikit-image", "scikit-learn",
        "tqdm", "ninja", "Pillow", "torchvision", "dill", "matplotlib", "tensorboard", "einops",
        "kornia", "lpips", "omegaconf", "PyYAML", "pandas", "imageio-ffmpeg", "protobuf", "termcolor", "yacs", "albumentations", "dlib",
        "huggingface_hub", "safetensors", "accelerate", "timm", "segmentation-models-pytorch", "torchmetrics", "psutil",
        "git+https://github.com/openai/CLIP.git",
    ])
    .run_commands(
        "git clone https://github.com/AIRI-Institute/HairFastGAN.git /app/HairFastGAN",
        "cd /app/HairFastGAN && git lfs install",
        "python -c \"import clip; clip.load('ViT-B/32', device='cpu')\"",
    )
)

volume = modal.Volume.from_name("hairfastgan-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

app = modal.App("hairfastgan-tryon", image=image)


# ── Model Ağırlıklarını İndir (Bir kere çalışır) ─────────────────────────────
@app.function(
    volumes={WEIGHTS_DIR: volume},
    timeout=600,
)
def download_weights():
    """Model ağırlıklarını Hugging Face'den indirir ve Volume'a kaydeder."""
    import subprocess
    import os
    weights_marker = Path(WEIGHTS_DIR) / "pretrained_models" / ".downloaded"
    if weights_marker.exists():
        print("✅ Ağırlıklar zaten mevcut, atlanıyor.")
        return
    print("📥 Model ağırlıkları indiriliyor...")
    subprocess.run(
        ["git", "clone", "https://huggingface.co/AIRI-Institute/HairFastGAN", "/tmp/hf_weights"],
        check=True,
    )
    subprocess.run(["git", "-C", "/tmp/hf_weights", "lfs", "pull"], check=True)
    subprocess.run(["cp", "-r", "/tmp/hf_weights/pretrained_models", f"{WEIGHTS_DIR}/pretrained_models"], check=True)
    subprocess.run(["cp", "-r", "/tmp/hf_weights/input", f"{WEIGHTS_DIR}/input"], check=True)
    weights_marker.parent.mkdir(parents=True, exist_ok=True)
    weights_marker.touch()
    volume.commit()
    print("✅ Ağırlıklar Volume'a kaydedildi.")


# ── Ana Swap Servisi (Class-Based / Warm Cache) ───────────────────────────────
@app.cls(
    gpu="T4",
    volumes={WEIGHTS_DIR: volume},
    timeout=900,
    scaledown_window=600,
)
class HairFastService:
    @modal.enter()
    def load_model(self):
        """
        Container ilk ayağa kalktığında (cold-start) bir kere çalışır.
        Tüm StyleGAN, StarGAN ve e4e modellerini GPU belleğine yükler.
        Böylece sonraki POST isteklerinde bağlantı kopmaz ve montaj 2 saniyede biter.
        """
        import sys
        import os
        import torch
        from pathlib import Path

        print("🚀 [Lifecycle] Derin öğrenme modelleri GPU belleğine yükleniyor...")
        sys.path.insert(0, "/app/HairFastGAN")

        hfgan_dir = Path("/app/HairFastGAN")
        weights_pretrained = Path(WEIGHTS_DIR) / "pretrained_models"
        
        # Sadece bir kere birim içindeki ana klasörleri oluştur (kırık symlink olmaması için)
        weights_pretrained.mkdir(parents=True, exist_ok=True)
        for subdir in ["StyleGAN", "StarGAN", "encoder4editing", "Alignment"]:
            (weights_pretrained / subdir).mkdir(exist_ok=True)

        target_pretrained = hfgan_dir / "pretrained_models"
        if not target_pretrained.exists():
            os.symlink(str(weights_pretrained), str(target_pretrained))

        # Modellerin göreceli yolları bulabilmesi için ana dizine geç
        os.chdir("/app/HairFastGAN")

        # Modeli başlat (Tüm alt ağlar burada yüklenir)
        from hair_swap import HairFast, get_parser
        args = get_parser().parse_args([])
        self.hair_fast = HairFast(args)
        print("✅ [Lifecycle] Modeller başarıyla GPU'ya yüklendi ve hazır!")

    @modal.fastapi_endpoint(method="POST", label="swap-hairstyle")
    def swap_hairstyle(self, data: dict):
        """
        POST isteği geldiğinde önceden yüklenmiş self.hair_fast modelini kullanır.
        """
        import io
        import base64
        import torch
        from PIL import Image
        import torchvision.transforms as T

        def decode_image(b64_str: str) -> Image.Image:
            return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

        def encode_image(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        def center_crop_resize(img: Image.Image, size: int = 1024) -> Image.Image:
            w, h = img.size
            crop = min(w, h)
            img = img.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
            return img.resize((size, size), Image.LANCZOS)

        to_pil = T.ToPILImage()

        user_img = center_crop_resize(decode_image(data["user_image"]))
        hairstyle_b64_list = data["hairstyle_images"]

        results = []
        errors = []

        for i, hs_b64 in enumerate(hairstyle_b64_list):
            try:
                hs_img = center_crop_resize(decode_image(hs_b64))
                
                # Zaten bellekte olan modeli kullanıyoruz
                result_tensor = self.hair_fast.swap(user_img, hs_img, user_img)

                if len(result_tensor.shape) == 4:
                    result_tensor = result_tensor.squeeze(0)
                if result_tensor.min() < 0:
                    result_tensor = (result_tensor + 1) / 2
                result_img = to_pil(result_tensor.clamp(0, 1))

                results.append(encode_image(result_img))
                errors.append("")
                print(f"✅ Hairstyle {i+1} başarıyla işlendi.")
            except Exception as e:
                results.append("")
                errors.append(str(e))
                print(f"❌ Hairstyle {i+1} hatası: {e}")
            finally:
                torch.cuda.empty_cache()

        return {"results": results, "errors": errors}


@app.local_entrypoint()
def main():
    print("Modal app hazır. Deploy etmek için: modal deploy modal_app.py")
