import unittest
import joblib
from score import score
import requests
import time
import subprocess

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

   def test_flask(self):
        # Test Flask integration
        server_process = subprocess.Popen(['python', 'app.py'])
        time.sleep(2)  # Wait for server to start

        url = 'http://localhost:5000/score'
        data = {'text': 'This is a test text.'}
        response = requests.post(url, json=data)
        prediction = response.json()['prediction']
        propensity = response.json()['propensity']

        server_process.terminate()
        server_process.wait()

        self.assertIn(prediction, [True, False])
        self.assertIsInstance(propensity, float)

if __name__ == "__main__":
   unittest.main()