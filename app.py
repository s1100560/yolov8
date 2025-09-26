import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
import warnings

# =========================
# ✅ PyTorch 2.6+ 相容性修復
# =========================
try:
    from torch.nn.modules.container import Sequential
    torch.serialization.add_safe_globals([Sequential])
    torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])
except Exception as e:
    print(f"⚠️ 相容性修復警告: {e}")
    warnings.filterwarnings("ignore")

# 建立 Flask app
app = Flask(__name__)

# =========================
# ✅ 載入模型
# =========================
try:
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
    print(f"載入模型從: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")

    # 強制用 weights_only=False 載入
    _ = torch.load(MODEL_PATH, weights_only=False)
    model = YOLO(MODEL_PATH)

    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None

# =========================
# ✅ API 路由
# =========================
@app.route("/")
def home():
    if model is None:
        return "❌ YOLOv8 API 啟動失敗：模型載入錯誤", 500
    return "✅ YOLOv8 API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型未正確載入"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        filepath = "temp.jpg"
        file.save(filepath)

        if not os.path.exists(filepath):
            return jsonify({"error": "檔案保存失敗"}), 500

        # 推論
        results = model(filepath)

        detections = []
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            for r in results[0].boxes:
                detections.append({
                    "class": model.names[int(r.cls)],
                    "confidence": float(r.conf),
                    "bbox": r.xyxy[0].tolist()
                })

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"detections": detections})

    except Exception as e:
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        return jsonify({"error": f"預測失敗: {str(e)}"}), 500

@app.route("/health")
def health():
    if model is None:
        return jsonify({"status": "error", "message": "模型未載入"}), 500
    return jsonify({"status": "healthy", "message": "服務正常"})

@app.route("/test")
def test():
    return jsonify({"message": "API 測試成功", "model_loaded": model is not None})


