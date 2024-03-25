from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the saved model
model_path = "./model/lightgbm_model.pkl"
model = joblib.load(model_path)

@app.route("/score", methods=["POST"])
def score_text():
    text = request.get_json()["text"]
    prediction, propensity = score(text, model, threshold=0.5)
    return jsonify({"prediction": prediction, "propensity": propensity})

if __name__ == "__main__":
    app.run(debug=True)