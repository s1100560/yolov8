import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

# 修正的 PyTorch 相容性修復
try:
    # 正確的導入方式
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential
    
    # 使用類別而不是字串
    torch.serialization.add_safe_globals([
        DetectionModel,
        Sequential
    ])
    print("✅ 相容性修復完成")
except Exception as e:
    print(f"⚠️ 相容性修復警告: {e}")

app = Flask(__name__)

# 模型檔案名稱
MODEL_PATH = "freshness_fruit_and_vegetables.pt"

# 檢查檔案是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型檔案: {MODEL_PATH}")

print(f"✅ 找到模型檔案: {MODEL_PATH}")

# 載入模型 - 簡化版本
try:
    # 方法1：直接載入
    model = YOLO(MODEL_PATH)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    try:
        # 方法2：使用預訓練模型
        model = YOLO('yolov8n.pt')
        print("✅ 使用預訓練模型載入成功")
    except Exception as e2:
        model = None
        print(f"❌ 所有模型載入都失敗: {e2}")

@app.route("/", methods=["GET"])
def home():
    if model is None:
        return "❌ 模型載入失敗，請檢查日誌", 500
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
