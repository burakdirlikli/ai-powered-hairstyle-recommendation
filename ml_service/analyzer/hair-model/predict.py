import cv2
import numpy as np
import joblib
import argparse
import sys
from skimage.feature import local_binary_pattern
import pywt
import os

# Feature Extraction Functions (MUST MATCH TRAIN.PY)
def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_wavelet_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = np.hstack([cA.ravel(), cH.ravel(), cV.ravel(), cD.ravel()])
    return features[:1024]  # reduce dimensionality

def extract_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def extract_features(img):
    color_hist = extract_color_histogram(img)
    lbp = extract_lbp(img)
    hu_moments = extract_hu_moments(img)
    wavelet = extract_wavelet_features(img)
    
    # Combine all features into a single vector
    return np.hstack([color_hist, lbp, hu_moments, wavelet])

def predict_hair_type(image_path, model_path=None, le_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(script_dir, 'hair_classifier.pkl')
    if le_path is None:
        le_path = os.path.join(script_dir, 'label_encoder.pkl')

    try:
        clf = joblib.load(model_path)
        le = joblib.load(le_path)
    except FileNotFoundError:
        return "Error: Model files not found. Train the model first."

    img = cv2.imread(image_path)
    if img is None:
        return f"Error: Could not read image at {image_path}"

    try:
        features = extract_features(img)
        features = features.reshape(1, -1)
        pred_encoded = clf.predict(features)
        
        # Get Probabilities
        probs = clf.predict_proba(features)[0]
        classes = le.classes_
        
        prediction = le.inverse_transform(pred_encoded)[0]
        
        # Create sorted probabilities list
        probs_dict = {classes[i]: probs[i] for i in range(len(classes))}
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

        return str(prediction), sorted_probs
    except Exception as e:
        return f"Error during prediction: {e}", []

def main():
    parser = argparse.ArgumentParser(description="Predict hair type from an image.")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    result = predict_hair_type(args.image_path)
    print(result)

if __name__ == "__main__":
    main()
