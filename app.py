import onnxruntime as ort
import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template
from PIL import Image
import io
import torch
from ultralytics.utils import ops
import base64

# 加載 ONNX 模型
session = ort.InferenceSession('freshness_fruit_and_vegetables.onnx')

app = Flask(__name__)

# 根據您的資料集類別（17個類別）
CLASS_NAMES = [
    "fresh apple", "fresh banana", "fresh bell pepper", "fresh carrot", "fresh cucumber",
    "fresh mango", "fresh orange", "fresh potato", "rotten apple", "rotten banana", 
    "rotten carrot", "rotten cucumber", "rotten mango", "rotten orange", "rotten potato",
    "rotten tomato", "rotten bell pepper"
]

@app.route("/")
def home():
    """首頁 - 網頁界面"""
    return render_template('index.html')

@app.route("/health")
def health_check():
    """健康檢查端點"""
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """API 預測端點"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "沒有上傳檔案"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "沒有選擇檔案"}), 400
        
        # 讀取和預處理影像
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        original_image = np.array(image)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        
        # 保存原始圖片用於顯示
        display_image = original_image.copy()
        
        # 預處理用於推理
        input_image = cv2.resize(image_rgb, (640, 640))
        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
        
        # 推理
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        results = session.run([output_name], {input_name: input_image})
        
        # 後處理
        predictions = torch.tensor(results[0])
        processed_results = ops.non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)
        
        # 提取檢測結果並繪製檢測框
        detections = []
        display_image_with_boxes = original_image.copy()
        
        for result in processed_results:
            if result is not None:
                for det in result:
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    
                    # 轉換座標回原始圖片尺寸
                    h, w = original_image.shape[:2]
                    x1 = int(x1 * w / 640)
                    y1 = int(y1 * h / 640)
                    x2 = int(x2 * w / 640)
                    y2 = int(y2 * h / 640)
                    
                    # 根據新鮮/腐爛選擇顏色
                    class_name = CLASS_NAMES[int(cls)]
                    if class_name.startswith('fresh'):
                        color = (0, 255, 0)  # 綠色 - 新鮮
                    else:
                        color = (0, 0, 255)  # 紅色 - 腐爛
                    
                    # 繪製檢測框
                    cv2.rectangle(display_image_with_boxes, (x1, y1), (x2, y2), color, 3)
                    
                    # 添加標籤
                    label = f"{class_name} {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_image_with_boxes, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(display_image_with_boxes, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    detections.append({
                        "class": int(cls),
                        "class_name": class_name,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                        "status": "fresh" if class_name.startswith('fresh') else "rotten"
                    })
        
        # 轉換為 base64 用於網頁顯示
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(display_image_with_boxes, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "fresh_count": len([d for d in detections if d["status"] == "fresh"]),
            "rotten_count": len([d for d in detections if d["status"] == "rotten"]),
            "image_with_boxes": f"data:image/jpeg;base64,{image_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
