import onnxruntime as ort
import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
import io

# 加載 ONNX 模型
session = ort.InferenceSession('freshness_fruit_and_vegetables.onnx')

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "ONNX YOLO API 正常運行", "status": "success"})

@app.route("/health")
def health_check():
    """健康檢查端點"""
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "沒有上傳檔案"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "沒有選擇檔案"}), 400
        
        # 讀取和預處理影像
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        # 推理
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        results = session.run([output_name], {input_name: image})
        
        return jsonify({
            "status": "success",
            "message": "推理完成",
            "output_shape": results[0].shape
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)  # 改為 8000
