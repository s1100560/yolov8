from flask import Flask, jsonify
import os
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 模型狀態
model_loaded = False
model = None

def safe_model_loader():
    """極度安全的模型載入方式"""
    global model, model_loaded
    
    try:
        logger.info("🔍 檢查模型檔案...")
        model_path = "freshness_fruit_and_vegetables.pt"
        
        if not os.path.exists(model_path):
            logger.error("❌ 模型檔案不存在")
            return False
            
        logger.info("✅ 模型檔案存在")
        
        # 檢查檔案大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"📁 模型檔案大小: {file_size:.2f} MB")
        
        # 嘗試導入 ultralytics（在獨立區塊中）
        try:
            logger.info("📦 導入 ultralytics...")
            from ultralytics import YOLO
        except ImportError as e:
            logger.error(f"❌ 導入失敗: {e}")
            return False
            
        # 嘗試載入模型
        try:
            logger.info("🚀 開始載入模型...")
            model = YOLO(model_path)
            logger.info("✅ 模型載入成功！")
            model_loaded = True
            return True
        except Exception as load_error:
            logger.error(f"❌ 模型載入錯誤: {load_error}")
            return False
            
    except Exception as e:
        logger.error(f"💥 嚴重錯誤: {e}")
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
    try:
        logger.info("🔄 收到模型載入請求")
        success = safe_model_loader()
        
        if success:
            return {
                "message": "模型載入成功",
                "model_loaded": True,
                "status": "success"
            }, 200
        else:
            return {
                "message": "模型載入失敗（請查看日誌）",
                "model_loaded": False,
                "status": "error"
            }, 500
            
    except Exception as e:
        logger.error(f"💥 載入端點錯誤: {e}")
        return {
            "message": "載入過程發生錯誤",
            "error": str(e),
            "status": "error"
        }, 500

@app.route("/model-info")
def model_info():
    """提供模型資訊（不觸發載入）"""
    model_path = "freshness_fruit_and_vegetables.pt"
    exists = os.path.exists(model_path)
    
    info = {
        "model_file_exists": exists,
        "model_loaded": model_loaded,
        "status": "info"
    }
    
    if exists:
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        info["file_size_mb"] = round(file_size, 2)
    
    return info

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
