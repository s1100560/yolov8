import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

# 完整的 PyTorch 2.8+ 相容性修復 - 添加所有必要的類別
try:
    # 導入所有需要的類別
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import Bottleneck, C2f
    from ultralytics.nn.modules.head import Detect
    from torch.nn.modules.container import Sequential
    from torch.nn import Conv2d, BatchNorm2d, SiLU, Upsample, MaxPool2d
    
    # 添加所有必要的安全全域變數
    torch.serialization.add_safe_globals([
        DetectionModel,
        Conv,
        Bottleneck,
        C2f,
        Detect,
        Sequential,
        Conv2d,
        BatchNorm2d,
        SiLU,
        Upsample,
        MaxPool2d
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

# 載入模型 - 使用環境變數繞過限制
try:
    # 設定環境變數，允許載入非 weights_only 的模型
    os.environ['PYTORCH_ENABLE_WEIGHTS_ONLY_LOAD'] = '0'
    
    # 方法1：直接載入
    model = YOLO(MODEL_PATH)
    print("✅ 自訂模型載入成功")
    
except Exception as e:
    print(f"❌ 自訂模型載入失敗: {e}")
    
    try:
        # 方法2：使用預訓練模型
        model = YOLO('yolov8n.pt')
        print("✅ 預訓練模型載入成功")
    except Exception as e2:
        model = None
        print(f"❌ 所有模型載入都失敗: {e2}")

@app.route("/", methods=["GET"])
def home():
    if model is None:
        return "⚠️ API 運行中，但模型載入失敗，請使用預訓練模型測試", 200
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
    return jsonify({"status": "success", "message": "API 正常運作", "model_type": "預訓練模型" if 'yolov8n' in str(model) else "自訂模型"})
