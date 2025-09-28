from flask import Flask, jsonify
import os
import torch
from torch.serialization import add_safe_globals

app = Flask(__name__)

# 修復 PyTorch 2.6 模型載入問題
def load_model_fixed(model_path):
    """修復 PyTorch 2.6 安全性限制的模型載入"""
    try:
        # 導入需要的類別
        from ultralytics.nn.tasks import DetectionModel
        from torch.nn.modules.container import Sequential
        
        # 添加安全全域變數（錯誤訊息要求的）
        add_safe_globals([DetectionModel, Sequential])
        
        print("🔧 使用安全全域變數載入模型...")
        
        # 使用 ultralytics 的 YOLO 載入（會自動處理兼容性）
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        print("✅ 模型載入成功！")
        return model, True
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return None, False

# 模型狀態
model_loaded = False
model = None

def initialize_model():
    """初始化模型"""
    global model, model_loaded
    
    model_path = "freshness_fruit_and_vegetables.pt"
    if os.path.exists(model_path):
        print(f"✅ 找到模型檔案: {model_path}")
        model, model_loaded = load_model_fixed(model_path)
    else:
        print("❌ 模型檔案不存在")
        model_loaded = False

@app.route("/")
def home():
    return {
        "message": "YOLOv8 API 運行中", 
        "model_loaded": model_loaded,
        "status": "ready"
    }

@app.route("/test")
def test():
    if model_loaded:
        return {"message": "模型已載入", "status": "success"}
    else:
        return {"message": "模型未載入", "status": "error"}

@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}, 200

@app.route("/load-model")
def load_model_endpoint():
    """手動觸發模型載入"""
    global model, model_loaded
    if not model_loaded:
        print("🔄 手動載入模型中...")
        initialize_model()
    
    if model_loaded:
        return {"message": "模型載入成功", "status": "success"}
    else:
        return {"message": "模型載入失敗", "status": "error"}, 500

@app.route("/predict", methods=["POST"])
def predict():
    global model, model_loaded
    
    # 第一次呼叫時才載入模型
    if not model_loaded:
        initialize_model()
    
    if not model_loaded:
        return {"error": "模型載入失敗", "status": "error"}, 500
    
    return {"message": "預測功能準備就緒", "status": "success"}

# 應用程式啟動時不自動載入模型，避免啟動失敗
# initialize_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

