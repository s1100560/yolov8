import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

# 正確的 PyTorch 2.8+ 相容性修復
try:
    # 方法1：使用新的 API
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"相容性修復警告: {e}")
    try:
        # 方法2：使用舊的 API 或全域設定
        torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])
    except:
        try:
            # 方法3：直接設定 weights_only=False
            import warnings
            warnings.filterwarnings("ignore")
        except:
            pass

# 建立 Flask app
app = Flask(__name__)

# 載入模型 - 添加錯誤處理
try:
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "freshness_fruit_and_vegetables.pt"))
    print(f"載入模型從: {MODEL_PATH}")
    
    # 檢查模型檔案是否存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案不存在: {MODEL_PATH}")
    
    # 載入模型
    model = YOLO(MODEL_PATH)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    # 創建一個空的模型變數，讓應用程式至少能啟動
    model = None

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
        # 存下暫存檔
        filepath = "temp.jpg"
        file.save(filepath)

        # 檢查檔案是否成功保存
        if not os.path.exists(filepath):
            return jsonify({"error": "檔案保存失敗"}), 500

        # 推論
        results = model(filepath)

        # 取第一張圖的結果
        detections = []
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            for r in results[0].boxes:
                detections.append({
                    "class": model.names[int(r.cls)],
                    "confidence": float(r.conf),
                    "bbox": r.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })

        # 清理暫存檔
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({"detections": detections})
        
    except Exception as e:
        # 清理暫存檔
        if os.path.exists("temp.jpg"):
            os.remove("temp.jpg")
        return jsonify({"error": f"預測失敗: {str(e)}"}), 500

@app.route("/health")
def health():
    """健康檢查端點"""
    if model is None:
        return jsonify({"status": "error", "message": "模型未載入"}), 500
    return jsonify({"status": "healthy", "message": "服務正常"})

# 添加一個簡單的測試端點
@app.route("/test")
def test():
    return jsonify({"message": "API 測試成功", "model_loaded": model is not None})



