"""
generator.py
------------
Genai tarafının ana public giriş noktası.

Sorumluluklar:
  - TryOnRequest'i alır
  - Her hairstyle için sıralı (sequential) üretim akışını yönetir
  - io_utils, prompts, schemas ve config modüllerini koordine eder
  - Her hairstyle için GenerationResult, run sonu için RunSummary üretir
  - Hata yönetimi hairstyle bazlı izole edilir (bir hata tüm run'ı çökertemez)

Şu an implementasyon içermiyor; bir sonraki adımda OpenAI entegrasyonu eklenecek.
"""

import os
import time
import dataclasses
from typing import Optional

from openai import OpenAI

from . import config as cfg
from .schemas import TryOnRequest, HairstyleInput, GenerationResult, RunSummary
from .prompts import build_prompt
from . import io_utils


class GenAITryOnGenerator:
    """
    Generative AI tabanlı visual try-on motoru.
    Provider: OpenAI

    Kullanım:
        generator = GenAITryOnGenerator()
        summary = generator.run(request)
    """

    def __init__(
        self,
        provider: str = cfg.PROVIDER,
        analysis_model: str = cfg.ANALYSIS_MODEL_NAME,
        generation_model: str = cfg.GENERATION_MODEL_NAME,
        output_root: str = cfg.OUTPUT_ROOT,
    ):
        self.provider = provider
        self.analysis_model = analysis_model
        self.generation_model = generation_model
        self.output_root = output_root

        # OpenAI client başlatma
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment değişkeni bulunamadı.")
        self.client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, request: TryOnRequest) -> RunSummary:
        """
        Ana giriş noktası. TryOnRequest alır, RunSummary döndürür.
        Her hairstyle sıralı olarak işlenir.
        """
        run_id = request.run_id or io_utils.generate_run_id()
        run_dir = io_utils.create_run_dir(self.output_root, run_id)

        print(f"\n[GenAI TryOn] Run başladı: {run_id}")
        print(f"[GenAI TryOn] Output: {run_dir}")

        results = []
        total_start = time.time()

        for i, hairstyle in enumerate(request.hairstyles, start=1):
            result = self._process_hairstyle(
                request=request,
                hairstyle=hairstyle,
                index=i,
                run_id=run_id,
                run_dir=run_dir,
            )
            results.append(result)

        total_latency = round(time.time() - total_start, 3)
        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful

        summary = RunSummary(
            run_id=run_id,
            method="genai",
            provider=self.provider,
            model_name=self.generation_model,
            prompt_version=request.prompt_version,
            user_image_path=request.user_image_path,
            total_hairstyles=len(results),
            successful=successful,
            failed=failed,
            total_latency_sec=total_latency,
            output_dir=run_dir,
            results=[dataclasses.asdict(r) for r in results],
            timestamp=io_utils.now_iso(),
        )

        # run_summary.json yaz
        summary_path = os.path.join(run_dir, "run_summary.json")
        io_utils.write_json(summary_path, dataclasses.asdict(summary))

        print(f"[GenAI TryOn] Run tamamlandı — Başarılı: {successful} / Başarısız: {failed}")
        return summary

    # ------------------------------------------------------------------
    # Per-Hairstyle İşlem
    # ------------------------------------------------------------------

    def _process_hairstyle(
        self,
        request: TryOnRequest,
        hairstyle: HairstyleInput,
        index: int,
        run_id: str,
        run_dir: str,
    ) -> GenerationResult:
        """
        Tek bir hairstyle için tam üretim akışını çalıştırır.
        Hata oluşursa GenerationResult içinde yakalar, tüm run'ı çökertmez.
        """
        print(f"[GenAI TryOn] [{index}] İşleniyor: {hairstyle.name}")

        hairstyle_dir = io_utils.create_hairstyle_dir(run_dir, index, hairstyle.name)
        ref_image_path = os.path.join(hairstyle_dir, "reference_hair.jpg")
        output_image_path = os.path.join(hairstyle_dir, "generated.png")
        metadata_path = os.path.join(hairstyle_dir, "metadata.json")

        base_prompt = build_prompt(version=request.prompt_version)
        start_time = time.time()

        try:
            # 1. Referans saç görselini indir
            io_utils.download_image(hairstyle.image_url, ref_image_path)

            # 2. Görsel üretimi (Vision -> DALL-E)
            gen_meta = self._generate_image(
                user_image_path=request.user_image_path,
                ref_image_path=ref_image_path,
                prompt_text=base_prompt,
                output_path=output_image_path
            )

            latency = round(time.time() - start_time, 3)
            result = GenerationResult(
                hairstyle_name=hairstyle.name,
                hairstyle_image_url=hairstyle.image_url,
                status="success",
                reference_image_path=ref_image_path,
                output_image_path=output_image_path,
                prompt_version=request.prompt_version,
                prompt_text=gen_meta.get("final_prompt", base_prompt),
                provider=self.provider,
                model_name=self.generation_model,
                latency_sec=latency,
                request_id=gen_meta.get("request_id"),
                raw_response_metadata=gen_meta
            )

        except Exception as e:
            latency = round(time.time() - start_time, 3)
            error_msg = str(e)
            print(f"[GenAI TryOn] [{index}] HATA: {error_msg}")

            result = GenerationResult(
                hairstyle_name=hairstyle.name,
                hairstyle_image_url=hairstyle.image_url,
                status="failed",
                reference_image_path=ref_image_path if os.path.exists(ref_image_path) else None,
                output_image_path=None,
                prompt_version=request.prompt_version,
                prompt_text=base_prompt,
                provider=self.provider,
                model_name=self.generation_model,
                latency_sec=latency,
                error_message=error_msg,
            )

        # metadata.json yaz
        io_utils.write_json(metadata_path, {
            "method": "genai",
            "run_id": run_id,
            "request_id": result.request_id,
            "provider": result.provider,
            "model_name": result.model_name,
            "hairstyle_name": result.hairstyle_name,
            "hairstyle_image_url": result.hairstyle_image_url,
            "user_image_path": request.user_image_path,
            "reference_image_path": result.reference_image_path,
            "output_image_path": result.output_image_path,
            "prompt_version": result.prompt_version,
            "prompt_text": result.prompt_text,
            "status": result.status,
            "latency_sec": result.latency_sec,
            "timestamp": io_utils.now_iso(),
            "raw_response_metadata": result.raw_response_metadata,
            "error_message": result.error_message,
        })

        return result

    # ------------------------------------------------------------------
    # Üretim Mantığı (Vision -> DALL-E)
    # ------------------------------------------------------------------

    def _generate_image(
        self,
        user_image_path: str,
        ref_image_path: str,
        prompt_text: str,
        output_path: str,
    ) -> dict:
        """
        Kullanıcı ve referans görsellerini analiz eder, 
        DALL-E 3 için optimize edilmiş bir prompt oluşturur ve görseli üretir.
        """
        
        # 1. GPT-4o Vision ile analiz ve prompt üretimi
        print(f"[GenAI TryOn] GPT-4o ile görseller analiz ediliyor...")
        user_base64 = io_utils.encode_image_to_base64(user_image_path)
        ref_base64 = io_utils.encode_image_to_base64(ref_image_path)

        vision_messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional image editor. Image A is the user, Image B is the hairstyle reference. "
                    "Your task is to write a precise DALL-E 3 prompt that describes a person who looks EXACTLY like Image A "
                    "but with the hairstyle from Image B. "
                    "Focus on physical consistency and high-quality photo details."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Instruction: {prompt_text}"},
                    {"type": "text", "text": "Image A (User):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_base64}"}},
                    {"type": "text", "text": "Image B (Hairstyle Reference):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_base64}"}},
                ]
            }
        ]

        vision_response = self.client.chat.completions.create(
            model=self.analysis_model,
            messages=vision_messages,
            max_tokens=500
        )
        
        final_dalle_prompt = vision_response.choices[0].message.content.strip()
        print(f"[GenAI TryOn] DALL-E 3 için optimize prompt üretildi.")

        # 2. DALL-E 3 ile üretim
        print(f"[GenAI TryOn] DALL-E 3 üretimi başlatılıyor...")
        dalle_response = self.client.images.generate(
            model=self.generation_model,
            prompt=final_dalle_prompt,
            size=cfg.IMAGE_SIZE,
            quality="hd",
            n=1,
            response_format="url"
        )

        generated_url = dalle_response.data[0].url
        
        # 3. Sonucu kaydet
        io_utils.download_image(generated_url, output_path)
        print(f"[GenAI TryOn] Üretilen görsel kaydedildi: {output_path}")

        return {
            "final_prompt": final_dalle_prompt,
            "vision_model": self.analysis_model,
            "generation_model": self.generation_model,
            "revised_prompt": dalle_response.data[0].revised_prompt if hasattr(dalle_response.data[0], 'revised_prompt') else None,
            "request_id": getattr(dalle_response, 'id', None)
        }
