from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)
CORS(app)

# ===============================
# 🔥 MODEL DOWNLOAD CONFIG
# ===============================
MODEL_PATH = "model.h5"
DRIVE_FILE_ID = "1HMF-GG0xEPlxYcI3wyW1p9YNkqRsSGtg"
MODEL_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# ===============================
# 🔽 DOWNLOAD MODEL IF NOT EXISTS
# ===============================
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading ML model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
else:
    print("✅ Model already exists, skipping download")

# ===============================
# LOAD MODEL & LABELS
# ===============================
print("📦 Loading TensorFlow model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

# ===============================
# IMAGE PREPROCESSING
# ===============================
def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "🌾 Sugarcane Disease AI API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    img = prepare_image(image_bytes)
    predictions = model.predict(img, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    return jsonify({
        "crop": "Sugarcane",
        "disease": class_names[predicted_index],
        "confidence": round(confidence, 2)
    })

# ===============================
# START SERVER (RENDER SAFE)
# ===============================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=PORT)
