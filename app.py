import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch

app = Flask(__name__)

MODEL_PATH = "freshness_fruit_and_vegetables.pt"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型檔案 {MODEL_PATH} 不存在")

    # 🔑 重點：避免 PyTorch 2.6 的 weights_only 限制
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    model = YOLO()
    model.model.load_state_dict(checkpoint["model"].state_dict() if "model" in checkpoint else checkpoint)
    print("✅ 模型載入成功")

except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model = None

