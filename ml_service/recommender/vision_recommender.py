import json
from openai import OpenAI

class VisionRecommender:
    """Uses OpenAI Vision to pick the best 5 hairstyles based on candidates & preferences."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        self.GITHUB_RAW_BASE = "https://raw.githubusercontent.com/burakdirlikli/hairstyle-catalog-images/main/images"

    def _get_image_url(self, hs_id: str, view: str = "front"):
        return f"{self.GITHUB_RAW_BASE}/{hs_id}_{view}.jpg"

    def recommend(self, face_shape: str, hair_type: str, candidates: dict, preferences: dict):
        print("[VisionRecommender] Calling OpenAI Vision API for top 5 selection...")
        # Token tasarrufu için sadece en yüksek puanlı 15 adayı API'ye gönderelim
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:15]
        
        system_instruction = (
            "You are an expert hair stylist. "
            "Select exactly 5 distinct hairstyle IDs from the provided candidates that best match the user's facial features, hair type, and preferences. "
            "Format Output: ONLY the 5 HS_IDs, one per line. No other text."
        )

        user_text = (
            f"Face: {face_shape}\n"
            f"Hair Type: {hair_type}\n"
            f"Preferences: {json.dumps(preferences)}\n\n"
            "Candidates (ID, Score):\n" +
            "\n".join([f"{cid}: {score}" for cid, score in sorted_candidates])
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ]

        # Sadece FRONT görsellerini 'low' detayla ekle ki token limitine (TPM) takılmayalım.
        for cid, score in sorted_candidates:
            url_front = self._get_image_url(cid, "front")
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": url_front, "detail": "low"}})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100
        )
        content = response.choices[0].message.content.strip()
        recommended_ids = [line.strip() for line in content.split('\n') if line.strip()]
        print(f"[VisionRecommender] OpenAI recommended: {recommended_ids}")
        return recommended_ids
