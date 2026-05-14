import os
import json
import sys
from dotenv import load_dotenv

# Ensure ml_service is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'ml_service'))

try:
    from analyzer import predict_all
    from recommender.candidate_selector import CandidateSelector
    from recommender.vision_recommender import VisionRecommender
    from virtual_try_on.manager import VirtualTryOnManager
except ImportError as e:
    print(f"Error importing ml_service modules: {e}")
    sys.exit(1)


class FaceHairAnalyzer:
    """Handles extracting face shape and hair type from an image using predict_all."""
    def analyze(self, image_path: str):
        print(f"[Analyzer] Analyzing image: {image_path}")
        result = predict_all.analyze_image(image_path)
        if "error" in result:
            raise ValueError(result["error"])
        
        analysis = result.get("analysis", {})
        
        # Mappings
        hair_type_map = {
            "straight": "straight",
            "wavy-curly": "wavy/curly",
            "buzz": "buzz"
        }
        face_shape_map = {
            "ovale": "ovale",
            "square": "square",
            "round": "round",
            "rectangular": "rectangular"
        }
        
        raw_face = analysis.get("face")
        raw_hair = analysis.get("hair")
        
        face_shape = face_shape_map.get(raw_face, raw_face) if raw_face else None
        hair_type = hair_type_map.get(raw_hair, raw_hair) if raw_hair else None
        
        print(f"[Analyzer] Result -> Face: {face_shape}, Hair: {hair_type}")
        return face_shape, hair_type


class AIEnginePipeline:
    """Orchestrates the entire end-to-end flow using modular components from ml_service."""
    def __init__(self):
        load_dotenv()
        self.analyzer = FaceHairAnalyzer()
        self.selector = CandidateSelector(os.path.join(current_dir, 'data', 'hs_definitions.csv'))
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            self.recommender = VisionRecommender(api_key=api_key) if api_key else None
        except Exception as e:
            print(f"[Warning] VisionRecommender not initialized: {e}")
            self.recommender = None
        self.try_on_manager = VirtualTryOnManager()
        
    def run(self, image_path: str, preferences: dict):
        print("\n=== AI Hairstyle Recommendation Engine Pipeline ===")
        # Step 1: Analyze Image
        face_shape, hair_type = self.analyzer.analyze(image_path)
        
        if not face_shape or not hair_type:
            print("Error: Could not determine face shape or hair type.")
            return None
            
        # Step 2: Get Candidates
        candidates = self.selector.get_candidates(face_shape, hair_type, min_score=3)
        if not candidates:
            print("No candidates found.")
            return None
            
        # Step 3: Recommend Top 5
        if self.recommender:
            top_5_ids = self.recommender.recommend(face_shape, hair_type, candidates, preferences)
        else:
            print("\n[Warning] Skipping VisionRecommender: OpenAI API key is missing. Using top 5 from Candidates instead.")
            # Fallback to just taking the top 5 scores
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            top_5_ids = [cid for cid, score in sorted_cands[:5]]
            
        # Step 4: Virtual Try-On (Artık prepare_sample.py / Backend tarafında, indirmeler bitince çağrılıyor)
        print("\n=== PIPELINE COMPLETED ===")
        print(f"Top 5 Recommendations: {top_5_ids}")

        return top_5_ids, None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Isolated AI Hairstyle Recommendation Engine')
    parser.add_argument('--image', type=str, required=True, help='Path to user photo')
    parser.add_argument('--prefs', type=str, required=False, help='Path to user preferences JSON file')
    args = parser.parse_args()
    
    prefs = {}
    if args.prefs and os.path.exists(args.prefs):
        with open(args.prefs, 'r', encoding='utf-8') as f:
            prefs = json.load(f)
    else:
        prefs = {
            "hair_length": "medium",
            "maintenance": "low",
            "beard": "yes",
            "usage": "casual"
        }
        
    pipeline = AIEnginePipeline()
    pipeline.run(args.image, prefs)
