from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown   # 🔥 ADD

app = Flask(__name__)
CORS(app)

# ===============================
# 🔥 DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================
MODEL_PATH = "model.h5"
DRIVE_FILE_ID = "1HMF-GG0xEPlxYcI3wyW1p9YNkqRsSGtg"  # ✅ YOUR FILE ID

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading ML model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={DRIVE_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 224

def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

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
    predictions = model.predict(img)[0]

    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index]) * 100

    return jsonify({
        "crop": "Sugarcane",
        "disease": class_names[predicted_index],
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=PORT)
