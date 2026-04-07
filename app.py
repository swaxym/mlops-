from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model1.pkl")


@app.route("/")
def home():
    return "ML Model API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
