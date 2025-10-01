import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['ULTRALYTICS_VERBOSE'] = 'False'

# 加載模型時阻止自動下載
from ultralytics import YOLO
model = YOLO('freshness_fruit_and_vegetables.pt', verbose=False)


from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "API 正常", "status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

