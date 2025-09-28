from flask import Flask, jsonify
import os

app = Flask(__name__)

# 模型狀態
model_loaded = False
model = None

def load_model_safely():
    """安全地載入模型"""
    global model, model_loaded
    
    try:
        print("🚀 開始載入模型...")
        
        # 檢查模型檔案是否存在
        model_path = "freshness_fruit_and_vegetables.pt"
        if not os.path.exists(model_path):
            print("❌ 模型檔案不存在")
            return False
            
        print("✅ 找到模型檔案")
        
        # 導入 ultralytics（在函數內導入，避免啟動時錯誤）
        from ultralytics import YOLO
        
        # 嘗試載入模型
        model = YOLO(model_path)
        print("✅ 模型載入成功！")
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        model_loaded = False
        return False

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
        "message": "模型未載入（延遲載入）" if not model_loaded else "模型已載入",
        "model_loaded": model_loaded,
        "status": "info" if not model_loaded else "success"
    }

@app.route("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}, 200

@app.route("/load-model")
def load_model_endpoint():
    """手動觸發模型載入"""
    success = load_model_safely()
    
    if success:
        return {
            "message": "模型載入成功",
            "model_loaded": True,
            "status": "success"
        }
    else:
        return {
            "message": "模型載入失敗",
            "model_loaded": False,
            "status": "error"
        }, 500

@app.route("/model-status")
def model_status():
    """檢查模型狀態"""
    return {
        "model_loaded": model_loaded,
        "status": "loaded" if model_loaded else "not_loaded"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)


