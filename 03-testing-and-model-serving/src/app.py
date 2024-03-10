from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the saved model
model_path = r"E:\Coding\Applied Machine Learning\03-testing-and-model-serving\model\lightgbm_model.pkl"
model = joblib.load(model_path)

@app.route("/score", methods=["POST"])
def score_text():
    text = request.json["text"]
    threshold = 0.5
    prediction, propensity = score(text, model, threshold)
    return jsonify({"prediction": bool(prediction), "propensity": float(propensity)})

if __name__ == "__main__":
    app.run(debug=True)