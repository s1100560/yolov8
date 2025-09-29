from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "YOLOv8 API 運行中", "status": "ready"}

@app.route("/test")
def test():
    return {"message": "API 測試成功", "status": "success"}

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
