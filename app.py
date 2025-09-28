from flask import Flask, jsonify
import os
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

app = Flask(__name__)

# 修復 PyTorch 2.6 模型載入問題
def load_model_with_fix(model_path):
    try:
        # 添加所有需要的安全全域變數
        add_safe_globals([DetectionModel, Sequential])
        
        print("🔧 使用安全全域變數設定...")
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ PyTorch 直接載入成功")
        
        # 如果是權重檔案，需要創建 YOLO 模型並載入權重
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # 先載入一個基礎模型
        
        if 'model' in model:
            yolo_model.model.load_state_dict(model['model'])
            print("✅ YOLO 權重載入成功")
            return yolo_model, True
        else:
            # 如果已經是完整的 YOLO 模型
            return YOLO(model_path), True
            
    except Exception as e:
        print(f"❌ 修復載入失敗: {e}")
        return None, False

def load_model_simple(model_path):
    """最簡單的載入方式，繞過安全性檢查"""
    try:
        # 方法1: 直接使用 YOLO 載入，但強制使用舊版載入方式
        from ultralytics import YOLO
        
        # 臨時修改環境變數，允許不安全載入
        os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'
        
        print("🚀 嘗試簡單載入...")
        model = YOLO(model_path)
        print("✅ 簡單載入成功")
        return model, True
        
    except Exception as e:
        print(f"❌ 簡單載入失敗: {e}")
        return None, False

# 載入模型
model_loaded = False
model = None

def initialize_model():
    global model, model_loaded
    
    model_path = "freshness_fruit_and_vegetables.pt"
    if os.path.exists(model_path):
        print("✅ 找到模型檔案:", model_path)
        
        # 嘗試多種載入方式
        model, success = load_model_simple(model_path)
        if not success:
            print("🔄 嘗試替代載入方式...")
            model, success = load_model_with_fix(model_path)
        
        model_loaded = success
    else:
        print("❌ 模型檔案不存在")
        model_loaded = False

# 應用程式啟動時不立即載入模型，避免啟動失敗
# initialize_model()  # 先註解掉，等需要時再載入

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
        return {"message": "模型未載入（延遲載入）", "status": "info"}

@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}, 200

@app.route("/load-model")
def load_model_endpoint():
    """手動觸發模型載入"""
    global model, model_loaded
    if not model_loaded:
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

if __name__ == "__main__":
    # 開發環境才立即載入模型
    if os.environ.get("ENV") == "development":
        initialize_model()
    
    app.run(host="0.0.0.0", port=10000, debug=False)
