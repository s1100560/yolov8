import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

# -------------------------------
# 初始化 Flask
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 模型載入
# -------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))

print(f"🔍 嘗試載入模型: {MODEL_PATH}")

model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")
    
    # ✅ 方案一：直接用 YOLO 載入，不使用 torch.load
    model = YOLO(MODEL_PATH)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")

# -------------------------------
# 首頁
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "🚀 YOLOv8 Flask API 已啟動", 200

# -------------------------------
# 推論 API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入"}), 500

    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片檔案"}), 400

    file = request.files["file"]
    img_path = os.path.join("/tmp", file.filename)
    file.save(img_path)

    try:
        results = model.predict(img_path)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()
                })
        return jsonify({"detections": detections})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# 主程式入口
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 預設會給 PORT 環境變數
    app.run(host="0.0.0.0", port=port)






