import os
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from torch.nn import Sequential
from torch.nn.modules.conv import Conv2d

# ✅ 強制允許 PyTorch 在載入 checkpoint 時用到的類別
torch.serialization.add_safe_globals([Sequential, Conv2d])
print("✅ Added Sequential and Conv2d to safe globals")

# Flask App
app = Flask(__name__)

# 嘗試載入模型
try:
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
    print(f"載入模型從: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")

    # 直接交給 YOLO 處理
    model = YOLO(MODEL_PATH)
    print("✅ 模型載入成功")

except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None


@app.route("/")
def home():
    return "🚀 YOLO Flask API Running!"


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入成功"}), 500

    if "file" not in request.files:
        return jsonify({"error": "請提供圖片檔案"}), 400

    file = request.files["file"]
    image_path = os.path.join("/tmp", file.filename)
    file.save(image_path)

    results = model(image_path)
    return jsonify(results[0].tojson())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)





