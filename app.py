from flask import Flask, jsonify  # 添加 jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "OK"})  # 使用 jsonify

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
