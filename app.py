from flask import Flask, jsonify
import os

app = Flask(__name__)

# 模型狀態
model_loaded = False
model = None

def load_model_safe(model_path):
    """安全的模型載入方式，避免啟動時崩潰"""
    try:
        print("🚀 嘗試載入模型...")
        
        # 方法1: 直接使用 ultralytics 的 YOLO
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ 模型載入成功")
        return model, True
        
    except Exception as e:
        print(f"❌ 載入失敗: {e}")
        
        # 方法2: 嘗試處理 PyTorch 2.6 安全性問題
        try:
            print("🔄 嘗試處理 PyTorch 2.6 兼容性...")
            import torch
            
            # 檢查是否有安全全域變數功能
            if hasattr(torch.serialization, 'add_safe_globals'):
                from ultralytics.nn.tasks import DetectionModel
                from torch.nn.modules.container import Sequential
                torch.serialization.add_safe_globals([DetectionModel, Sequential])
            
            # 重新載入
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("✅ 兼容性載入成功")
            return model, True
            
        except Exception as e2:
            print(f"❌ 所有載入方式都失敗: {e2}")
            return None, False

def initialize_model():
    """初始化模型"""
    global model, model_loaded
    
    model_path = "freshness_fruit_and_vegetables.pt"
    if os.path.exists(model_path):
        print(f"✅ 找到模型檔案: {model_path}")
        model, model_loaded = load_model_safe(model_path)
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
    return {
        "message": "基本 API 測試成功", 
        "model_loaded": model_loaded,
        "status": "success"
    }

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
    
    return {
        "message": "模型載入成功" if model_loaded else "模型載入失敗",
        "model_loaded": model_loaded,
        "status": "success" if model_loaded else "error"
    }

# 重要：啟動時不自動載入模型
# initialize_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

