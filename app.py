from flask import Flask, jsonify
from ultralytics import YOLO
import os
import torch

app = Flask(__name__)

# 修復 PyTorch 2.6 模型載入問題
def load_model_safely(model_path):
    try:
        # 方法1: 使用 weights_only=False（信任來源時使用）
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        print(f"標準載入失敗: {e}")
        try:
            # 方法2: 使用安全全域變數設定
            from torch.serialization import add_safe_globals
            from ultralytics.nn.tasks import DetectionModel
            
            # 添加安全全域變數
            add_safe_globals([DetectionModel])
            
            # 重新載入模型
            model = YOLO(model_path)
            return model, True
        except Exception as e2:
            print(f"安全載入也失敗: {e2}")
            return None, False

# 載入模型
model_path = "freshness_fruit_and_vegetables.pt"
if os.path.exists(model_path):
    print("✅ 找到模型檔案:", model_path)
    model, load_success = load_model_safely(model_path)
    if load_success:
        print("✅ 模型載入成功")
        model_loaded = True
    else:
        print("❌ 模型載入失敗")
        model_loaded = False
else:
    print("❌ 模型檔案不存在")
    model_loaded = False

@app.route("/")
def home():
    return {"message": "YOLOv8 API 運行中", "model_loaded": model_loaded}

@app.route("/test")
def test():
    if model_loaded:
        return {"message": "模型已載入", "status": "success"}
    else:
        return {"message": "模型未載入", "status": "error"}

@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return {"error": "模型未載入", "status": "error"}, 500
    
    # 這裡添加你的預測邏輯
    return {"message": "預測功能準備就緒", "status": "success"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
