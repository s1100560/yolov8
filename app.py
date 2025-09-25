import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

# 修復 PyTorch 2.8+ 相容性問題
try:
    torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])
except:
    pass

# 建立 Flask app
app = Flask(__name__)

# 載入模型
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
model = YOLO(MODEL_PATH)

@app.route("/")
def home():
    return "✅ YOLOv8 API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # 存下暫存檔
    filepath = "temp.jpg"
    file.save(filepath)

    # 推論
    results = model(filepath)

    # 取第一張圖的結果
    detections = []
    for r in results[0].boxes:
        detections.append({
            "class": model.names[int(r.cls)],
            "confidence": float(r.conf),
            "bbox": r.xyxy[0].tolist()  # [x1, y1, x2, y2]
        })

    return jsonify({"detections": detections})





