import pandas as pd

class CandidateSelector:
    """Reads scores from CSV and selects top candidates based on face/hair analysis."""
    def __init__(self, data_path: str):
        self.csv_path = data_path
        
    def get_candidates(self, face_shape: str, hair_type: str, min_score: int = 3):
        print(f"[CandidateSelector] Filtering candidates for {face_shape} & {hair_type} (score >= {min_score})")
        df = pd.read_csv(self.csv_path)
        filtered = df[(df['face_analysis'] == face_shape) & 
                      (df['hair_analysis'] == hair_type) & 
                      (df['score'] >= min_score)]
        
        scores = {}
        for _, row in filtered.iterrows():
            scores[row['hs_id']] = row['score']
            
        print(f"[CandidateSelector] Found {len(scores)} candidates.")
        return scores
