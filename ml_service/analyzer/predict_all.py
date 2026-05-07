import sys
import os
import importlib.util
import json

# Add model directories to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'face-model'))
sys.path.append(os.path.join(current_dir, 'hair-model'))

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Lazy loading - modüller sadece analyze_image çağrıldığında yüklenecek
face_predict = None
hair_predict = None
face_model_dir = None
hair_model_dir = None

def _load_models():
    """Modelleri lazy olarak yükle"""
    global face_predict, hair_predict, face_model_dir, hair_model_dir
    
    if face_predict is None or hair_predict is None:
        try:
            # Load Face Predictor
            face_model_dir = os.path.join(current_dir, 'face-model')
            face_predict = load_module("face_predict", os.path.join(face_model_dir, "predict.py"))

            # Load Hair Predictor
            hair_model_dir = os.path.join(current_dir, 'hair-model')
            hair_predict = load_module("hair_predict", os.path.join(hair_model_dir, "predict.py"))
        except Exception as e:
            # sys.exit yerine exception fırlat - worker çökmesin
            raise ImportError(f"Error importing ML modules: {e}")

def analyze_image(image_path):
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}

    # Modelleri lazy olarak yükle
    try:
        _load_models()
    except ImportError as e:
        return {"error": str(e)}

    face_label = None
    hair_label = None

    # --- Face Shape Prediction ---
    try:
        if face_model_dir not in sys.path:
             sys.path.append(face_model_dir)

        face_label, face_probs = face_predict.predict_single(image_path)
    except Exception as e:
        # Hata durumunda None döndür, çökme
        pass

    # --- Hair Type Prediction ---
    try:
        if hair_model_dir not in sys.path:
             sys.path.append(hair_model_dir)
             
        hair_label, hair_probs = hair_predict.predict_hair_type(image_path)
    except Exception as e:
        # Hata durumunda None döndür, çökme
        pass
    
    # --- Final Result ---
    # Not: Skorlar artık database'den çekilecek (tasks.py içinde)
    # Bu fonksiyon sadece yüz ve saç tipi analizi yapar
    final_output = {
        "analysis": {
            "face": str(face_label) if face_label else None,
            "hair": str(hair_label) if hair_label else None
        }
    }
    
    return final_output

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python predict_all.py <path_to_image>"}))
        return

    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
