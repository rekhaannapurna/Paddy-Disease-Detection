import io
import os
import sys
import types
import pickle
import pathlib
import logging
from typing import Tuple, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# File Upload Limit (10MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# CORS Setup
frontend_url = os.environ.get("FRONTEND_URL", "*")
CORS(app, resources={r"/*": {"origins": frontend_url}})

# Constants & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOBILENET_PKL_PATH = os.path.join(BASE_DIR, "Models", "MobileNetV2_paddy_baseline.pkl")
KNN_PKL_PATH = os.path.join(BASE_DIR, "Models", "MobileNetV2_models", "KNN_model.pkl")

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]

# MobileNetV2 Feature Extractor Module (matching Colab/MoblieNetV2_features.ipynb Cell 16)
class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

# Global Model References
feature_extractor = None
knn_model = None
models_loaded_successfully = False

def _setup_unpickler_compatibility():
    """Mock missing modules and compatibility shims for loading fastai exported .pkl models."""
    sys.modules["pathlib._local"] = pathlib
    pathlib._local = pathlib

    def make_mock(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            sys.modules[name] = m

    make_mock("fasttransform")
    make_mock("fasttransform.transform")

    class DummyPipeline:
        pass

    class DummyTransform:
        pass

    sys.modules["fasttransform.transform"].Pipeline = DummyPipeline
    sys.modules["fasttransform.transform"].Transform = DummyTransform

    plum_mods = ["plum", "plum._function", "plum._resolver", "plum._method", "plum._signature", "plum._util"]
    for pm in plum_mods:
        make_mock(pm)
        m = sys.modules[pm]
        setattr(m, "Function", type("Function", (), {}))
        setattr(m, "Resolver", type("Resolver", (), {}))
        setattr(m, "MethodList", type("MethodList", (list,), {}))
        setattr(m, "Method", type("Method", (), {}))
        setattr(m, "Signature", type("Signature", (), {}))
        setattr(m, "Missing", type("Missing", (), {}))

class SafePyUnpickler(pickle._Unpickler):
    """Custom unpickler to handle Python version differences (e.g. CodeType arguments)."""
    def __init__(self, file, *args, **kwargs):
        kwargs.pop("map_location", None)
        kwargs.pop("encoding", None)
        super().__init__(file)

    def load_reduce(self):
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        try:
            stack[-1] = func(*args)
        except Exception:
            stack[-1] = None

SafePyUnpickler.dispatch = pickle._Unpickler.dispatch.copy()
SafePyUnpickler.dispatch[ord("R")] = SafePyUnpickler.load_reduce
SafePyUnpickler.dispatch[ord("o")] = SafePyUnpickler.load_reduce

class CustomPickleModule:
    Unpickler = SafePyUnpickler
    load = pickle.load
    UnpicklingError = pickle.UnpicklingError

def load_models():
    """Loads MobileNetV2 from MobileNetV2_paddy_baseline.pkl and KNN_model.pkl once on startup."""
    global feature_extractor, knn_model, models_loaded_successfully

    logger.info("Initializing models...")
    _setup_unpickler_compatibility()

    try:
        # 1. Load MobileNetV2 Learner from MobileNetV2_paddy_baseline.pkl
        logger.info(f"Loading MobileNetV2 feature extractor from: {MOBILENET_PKL_PATH}")
        with open(MOBILENET_PKL_PATH, "rb") as f:
            learner = torch.load(f, map_location="cpu", weights_only=False, pickle_module=CustomPickleModule)

        base_model = learner.model
        base_model.eval()

        feature_extractor = MobileNetV2FeatureExtractor(base_model)
        feature_extractor.eval()

        # 2. Load KNN Model from KNN_model.pkl
        logger.info(f"Loading KNN model from: {KNN_PKL_PATH}")
        knn_model = joblib.load(KNN_PKL_PATH)

        models_loaded_successfully = True
        logger.info("✅ All models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}", exc_info=True)
        models_loaded_successfully = False

# Execute model loading when app starts
load_models()

# Exact Preprocessing Pipeline (matching Colab/MoblieNetV2_features.ipynb)
# Image -> RGB -> Resize(480) squish -> Resize(224) -> ToTensor -> ImageNet Normalize
image_transform = T.Compose([
    T.Resize((480, 480), interpolation=T.InterpolationMode.BILINEAR),
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Paddy Disease Detection API",
        "version": "1.0.0",
        "status": "online",
        "health_check": "/health",
        "prediction_endpoint": "POST /predict"
    })

@app.route("/health", methods=["GET"])
def health():
    if not models_loaded_successfully:
        return jsonify({
            "status": "error",
            "message": "Models failed to load on startup",
            "models_loaded": False
        }), 503

    return jsonify({
        "status": "ok",
        "model": "MobileNetV2 (.pkl) + KNN",
        "models_loaded": True
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded_successfully:
        return jsonify({"success": False, "error": "Backend models are not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request field 'image'"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "Empty file provided"}), 400

    try:
        # Read image in memory
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verify it's a valid image
        
        # Re-open after verify as per PIL docs
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning(f"Invalid image uploaded: {str(e)}")
        return jsonify({"success": False, "error": "Invalid or corrupted image file"}), 400

    try:
        # Preprocessing
        tensor_img = image_transform(pil_img).unsqueeze(0)  # Shape (1, 3, 224, 224)

        # Feature Extraction
        with torch.no_grad():
            features = feature_extractor(tensor_img).cpu().numpy()  # Shape (1, 1280)

        if features.shape != (1, 1280):
            logger.error(f"Unexpected feature shape: {features.shape}")
            return jsonify({"success": False, "error": f"Feature extraction output shape error: {features.shape}"}), 500

        # KNN Inference
        class_id = int(knn_model.predict(features)[0])
        probabilities = knn_model.predict_proba(features)[0]
        confidence = float(probabilities[class_id])

        prediction_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}"

        return jsonify({
            "success": True,
            "prediction": prediction_name,
            "class_id": class_id,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": "An error occurred during image inference"}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"success": False, "error": "File size exceeds 10MB limit"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
