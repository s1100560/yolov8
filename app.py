from flask import Flask, jsonify
import os
import sys

app = Flask(__name__)

# 先不載入模型，測試基本功能
model_loaded = False
model = None

@app.route("/")
def home():
    return {
        "message": "YOLOv8 API 運行中", 
        "model_loaded": model_loaded,
        "status": "basic_test"
    }

@app.route("/test")
def test():
    return {
        "message": "基本測試成功", 
        "model_loaded": model_loaded,
        "status": "success"
    }

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

def load_model_later():
    """在需要時才載入模型"""
    global model, model_loaded
    try:
        from ultralytics import YOLO
        print("🚀 開始載入模型...")
        model = YOLO("freshness_fruit_and_vegetables.pt")
        model_loaded = True
        print("✅ 模型載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        model_loaded = False

@app.route("/predict", methods=["POST"])
def predict():
    global model, model_loaded
    
    # 第一次呼叫時才載入模型
    if not model_loaded:
        load_model_later()
    
    if not model_loaded:
        return {"error": "模型載入失敗", "status": "error"}, 500
    
    return {"message": "預測功能準備就緒", "status": "success"}

if __name__ == "__main__":
    # 開發時才立即載入模型
    if os.environ.get("ENV") == "development":
        load_model_later()
    
    app.run(host="0.0.0.0", port=10000, debug=False)
