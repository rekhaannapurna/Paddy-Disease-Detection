# Paddy Disease Detection Backend API

Production-ready Flask backend REST API for Paddy Disease Detection using a fine-tuned MobileNetV2 feature extractor and a pre-trained K-Nearest Neighbors (KNN) classifier (`KNN_model.pkl`).

---

## 🏗️ Architecture & Complete Inference Pipeline

```
Frontend Image Upload
        │
        ▼
POST /predict (multipart/form-data)
        │
        ▼
Backend API (app.py)
        │
        ▼
Image Preprocessing (PIL RGB → Resize 480x480 squish → Resize 224x224 → ImageNet Normalization)
        │
        ▼
MobileNetV2 Feature Extraction (Loaded from MobileNetV2_paddy_baseline.pkl)
        │
        ▼
1280-Dimensional Feature Vector (1, 1280)
        │
        ▼
KNN Model (KNN_model.pkl fit with 8,326 training samples, k=5, Euclidean metric)
        │
        ▼
Disease Class Prediction & Confidence Calculation
        │
        ▼
JSON Response to Frontend
```

---

## 🚀 Frontend Integration Guide

### Endpoint
`POST /predict`

### Request Format
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `image` (the raw image file from file input)

> **IMPORTANT**: Do NOT manually set the `Content-Type` header when sending `FormData`. The browser will automatically set the header along with the correct multipart boundary.

### JavaScript `fetch` Example
```javascript
const sendImageForPrediction = async (file) => {
    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("https://YOUR-RAILWAY-BACKEND.up.railway.app/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            console.log("Predicted Disease:", result.prediction);
            console.log("Class ID:", result.class_id);
            console.log("Confidence Score:", result.confidence);
        } else {
            console.error("Prediction Error:", result.error);
        }
    } catch (error) {
        console.error("Network error:", error);
    }
};
```

### Successful Response Format (`HTTP 200`)
```json
{
  "success": true,
  "prediction": "bacterial_leaf_blight",
  "class_id": 0,
  "confidence": 0.9425
}
```

### Error Response Format (`HTTP 400 / 413 / 503`)
```json
{
  "success": false,
  "error": "No image file provided in request field 'image'"
}
```

---

## 🏥 Health Endpoint

### Endpoint
`GET /health`

### Response (`HTTP 200`)
```json
{
  "status": "ok",
  "model": "MobileNetV2 (.pkl) + KNN",
  "models_loaded": true
}
```

---

## 🌾 Supported Disease Classes (IDs 0–9)

| Class ID | Disease Name |
| :--- | :--- |
| `0` | `bacterial_leaf_blight` |
| `1` | `bacterial_leaf_streak` |
| `2` | `bacterial_panicle_blight` |
| `3` | `blast` |
| `4` | `brown_spot` |
| `5` | `dead_heart` |
| `6` | `downy_mildew` |
| `7` | `hispa` |
| `8` | `normal` |
| `9` | `tungro` |

---

## 🚂 Railway Deployment Instructions (Recommended)

Railway is extremely fast, reliable, and handles PyTorch/ML deployments with zero friction.

1. Go to [railway.app](https://railway.app) and sign in.
2. Click **+ New Project** $\rightarrow$ Select **Deploy from GitHub repo**.
3. Choose the repository: `rekhaannapurna/Paddy-Disease-Detection`.
4. Railway will automatically detect `railway.json`, `Procfile`, and `requirements.txt` and start the deployment.
5. In your Railway service settings under **Networking**, click **Generate Domain** to get your public backend URL (e.g. `https://paddy-disease-detection.up.railway.app`).
6. (Optional) In **Variables**, add `FRONTEND_URL` with your friend's frontend domain (default is `*`).

---

## 💻 Local Setup & Development

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Server (Development)
```bash
python3 app.py
```

### 3. Run API Server (Gunicorn Production Mode)
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```
