import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

# 建立 Flask app
app = Flask(__name__)

# 載入 YOLOv8 模型
MODEL_PATH = "freshness_fruit_and_vegetables.pt"
print(f"🔄 載入模型從: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)  # ✅ 官方 API，自動處理
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 YOLOv8 Flask API 運行中！"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未成功載入"}), 500

    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片 (form-data key = file)"}), 400

    file = request.files["file"]

    try:
        # 讀取圖片
        img = Image.open(io.BytesIO(file.read()))

        # 推論
        results = model.predict(img)

        # 把 YOLO 輸出整理成 JSON
        predictions = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = model.names[cls_id] if model.names else str(cls_id)
                predictions.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "box": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 預設使用 10000
    app.run(host="0.0.0.0", port=port)





