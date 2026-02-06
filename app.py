from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ===============================
# 🔥 LOAD TFLITE MODEL
# ===============================
MODEL_PATH = "model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

# ===============================
# LOAD LABELS
# ===============================
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ===============================
# IMAGE PREPROCESSING
# ===============================
def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# ROUTES
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "🌾 Sugarcane Disease AI API (TFLite) Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    img = prepare_image(image_bytes)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(predictions))

    return jsonify({
        "crop": "Sugarcane",
        "disease": class_names[idx],
        "confidence": round(float(predictions[idx]) * 100, 2)
    })

# ===============================
# START SERVER (RENDER SAFE)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
