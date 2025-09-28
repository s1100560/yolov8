from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# 模型檔案名稱
MODEL_PATH = "freshness_fruit_and_vegetables.pt"

# 檢查檔案是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 找不到模型檔案: {MODEL_PATH}")
    model = None
else:
    print(f"✅ 找到模型檔案: {MODEL_PATH}")
    try:
        # 直接載入模型
        model = YOLO(MODEL_PATH)
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        model = None

@app.route("/", methods=["GET"])
def home():
    if model is None:
        return "⚠️ API 運行中，但模型載入失敗，請檢查 /test 端點", 200
    return "🚀 YOLOv8 Flask API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入成功"}), 500

    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片檔案"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "沒有選擇檔案"}), 400

    try:
        # 暫存圖片
        filepath = "temp_image.jpg"
        file.save(filepath)

        # 模型推論
        results = model(filepath)
        detections = []
        
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

        # 清理暫存檔
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"detections": detections})
        
    except Exception as e:
        # 清理暫存檔
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        return jsonify({"error": f"預測失敗: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/test", methods=["GET"])
def test():
    """測試端點"""
    if model is None:
        return jsonify({"status": "error", "message": "模型未載入"})
    return jsonify({"status": "success", "message": "API 正常運作"})
