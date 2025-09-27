import os
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ⚠️ 改這裡：模型名稱要和你 repo 裡的一模一樣
MODEL_PATH = "freshness_fruit_and_vegetables.pt"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案 {MODEL_PATH} 不存在，請確認已放在專案根目錄並 push 到 GitHub。")
    
    model = YOLO(MODEL_PATH)
    print("✅ 模型載入成功")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "YOLOv8 Flask API 運行中 🚀"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未載入"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "請上傳圖片檔案，key 必須是 'file'"}), 400

    file = request.files["file"]

    # 存檔
    save_path = "upload.jpg"
    file.save(save_path)

    # 推論
    results = model(save_path)

    # 取 YOLO 偵測結果
    predictions = []
    for box in results[0].boxes:
        predictions.append({
            "class": int(box.cls),
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    return jsonify({
        "status": "success",
        "predictions": predictions
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

