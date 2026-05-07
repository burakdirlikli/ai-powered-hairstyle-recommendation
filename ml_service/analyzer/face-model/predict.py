import os
import sys
import warnings
import contextlib

# Context manager to suppress C-level stderr (for MediaPipe/TensorFlow logs)
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        try:
            old_stderr_fd = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            try:
                yield
            finally:
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stderr_fd)
        except Exception:
            # Fallback if file descriptors are not accessible (e.g. in some IDEs)
            yield

# Suppress TensorFlow/ABSL Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import joblib
import numpy as np
import pandas as pd

# Import feature_extractor normally (lazy init prevents logs here)
import feature_extractor

# Suppress other warnings
warnings.filterwarnings('ignore')

# Constants (Resolved relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_shape_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")

# Feature Columns (Must match training order exactly)
FEATURE_COLS = [
    'FW', 'FH', 'CW', 'JW', 'MFH', 'LFH', 'R1', 'R2', 
    'A_L1', 'A_R1', 'A_L2', 'A_R2', 'A_mean', 'A_var', 
    'J_dev_L', 'J_dev_R', 'J_dev_mean', 'J_dev_diff', 
    'ChinSharp_L', 'ChinSharp_R', 
    'R3', 'R4', 
    'Side_dev_L', 'Side_dev_R', 'Side_dev_mean'
]

def predict_single(image_path):
    # 1. Load Artifacts
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Run train_model.py first.")
        return
        
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    
    # 2. Process Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    # Lazy init triggers here, suppress logs
    with suppress_stderr():
        lm = feature_extractor.get_landmarks_raw(image_path)
        
    if lm is None:
        print("Error: Could not detect face or image quality too low.")
        return
        
    norm_lm = feature_extractor.normalize_pose(lm)
    feats_dict = feature_extractor.extract_features(norm_lm)
    
    # 3. Create DataFrame
    # Ensure order is correct
    df = pd.DataFrame([feats_dict])
    df = df[FEATURE_COLS]
    
    # 4. Scale
    X_scaled = scaler.transform(df)
    
    # 5. Predict
    pred_idx = clf.predict(X_scaled)[0]
    probs = clf.predict_proba(X_scaled)[0]
    try:
        pred_label = le.inverse_transform([pred_idx])[0]
        classes = le.classes_
    except:
        # Fallback if LE has issues, but it shouldn't
        classes = clf.classes_ # might be indices
        pred_label = str(pred_idx)
    
    # 6. Output
    # 6. Output
    # Create sorted probabilities list
    probs_dict = {classes[i]: probs[i] for i in range(len(classes))}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    
    return str(pred_label), sorted_probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        label, probs = predict_single(sys.argv[1])
        print(f"\nPredicted label: {label}\n")
        for cls, prob in probs:
            print(f"{cls}: {prob:.4f}")
