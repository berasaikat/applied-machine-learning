import os
import unittest
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from score import score
import multiprocessing
import requests
import os
import time
import signal
from flask import Flask, jsonify, request
from score import score
import threading

class TestScoring(unittest.TestCase):
   def setUp(self):
       # Load the saved model
       self.model_path = r"E:\Coding\Applied Machine Learning\03-testing-and-model-serving\model\lightgbm_model.pkl"
       self.model = joblib.load(self.model_path)
       self.spam_text = "Free money! Click here to get rich quick!"
       self.non_spam_text = "Hello, how are you doing today?"

   def test_score_output(self):
       # Test if the function produces some output without crashing
       text = "This is a test text."
       threshold = 0.5
       prediction, propensity = score(text, self.model, threshold)
       self.assertIsNotNone(prediction)
       self.assertIsNotNone(propensity)

   def test_score_format(self):
       # Test if the input/output formats/types are as expected
       text = "This is a test text."
       threshold = 0.5
       prediction, propensity = score(text, self.model, threshold)
       self.assertIsInstance(prediction, bool)
       self.assertIsInstance(propensity, float)

   def test_prediction_value(self):
       # Test if the prediction value is 0 or 1
       text = "This is a test text."
       threshold = 0.5
       prediction, _ = score(text, self.model, threshold)
       self.assertIn(prediction, [True, False])

   def test_propensity_range(self):
       # Test if the propensity score is between 0 and 1
       text = "This is a test text."
       threshold = 0.5
       _, propensity = score(text, self.model, threshold)
       self.assertGreaterEqual(propensity, 0)
       self.assertLessEqual(propensity, 1)

   def test_threshold_0(self):
       # Test if the prediction is always 1 when threshold is 0
       text = "This is a test text."
       threshold = 0
       prediction, _ = score(text, self.model, threshold)
       self.assertTrue(prediction)

   def test_threshold_1(self):
       # Test if the prediction is always 0 when threshold is 1
       text = "This is a test text."
       threshold = 1
       prediction, _ = score(text, self.model, threshold)
       self.assertFalse(prediction)

   def test_spam_text(self):
       # Test if the prediction is 1 for an obvious spam text
       threshold = 0.5
       prediction, _ = score(self.spam_text, self.model, threshold)
       self.assertTrue(prediction)

   def test_non_spam_text(self):
       # Test if the prediction is 0 for an obvious non-spam text
       threshold = 0.5
       prediction, _ = score(self.non_spam_text, self.model, threshold)
       self.assertFalse(prediction)

   def test_flask_app(self):
        # Create an event for stopping the Flask thread
        stop_event = threading.Event()

        # Start the Flask app in a separate thread
        flask_thread = threading.Thread(target=self.run_flask_app, args=(stop_event,))
        flask_thread.start()

        try:
            # Wait for the app to start
            time.sleep(2)

            # Test the /score endpoint
            text = "This is a test text."
            url = "http://localhost:5000/score"
            response = requests.post(url, json={"text": text})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("prediction", data)
            self.assertIn("propensity", data)
        finally:
            # Signal the Flask thread to stop
            stop_event.set()
            flask_thread.join()

   def run_flask_app(self, stop_event):
        # Load the saved model
        model_path = r"E:\Coding\Applied Machine Learning\03-testing-and-model-serving\model\lightgbm_model.pkl"
        model = joblib.load(model_path)

        # Create the Flask app
        app = Flask(__name__)

        @app.route("/score", methods=["POST"])
        def score_text():
            text = request.json["text"]
            threshold = 0.5
            prediction, propensity = score(text, model, threshold)
            return jsonify({"prediction": bool(prediction), "propensity": float(propensity)})

        # Run the Flask app until the stop_event is set
        while not stop_event.is_set():
            app.run(debug=False, use_reloader=False)

if __name__ == "__main__":
   unittest.main()