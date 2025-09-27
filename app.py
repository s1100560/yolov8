from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# 模型檔案名稱（放在 repo 根目錄，並且有 commit 到 GitHub）
MODEL_PATH = "freshness_fruit_and_vegetables.pt"

# 檢查檔案是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型檔案: {MODEL_PATH}")

# 載入模型
try:
    model = YOLO(MODEL_PATH)  # 不需要 weights_only
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return "🚀 YOLOv8 Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入成功"}), 500

    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片檔案"}), 400

    file = request.files["file"]

    # 暫存圖片
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    # 模型推論
    try:
        results = model(filepath)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

        return jsonify({"detections": detections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 本地測試時用，Render 會自動用 gunicorn
    app.run(host="0.0.0.0", port=5000)

