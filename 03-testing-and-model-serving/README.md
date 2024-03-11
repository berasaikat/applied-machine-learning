# README.md

## Overview

This repository contains code for a machine learning model testing and serving project. The project involves unit testing of the scoring function and integration testing of a Flask endpoint for serving model predictions.      
The model used in this project is a LightGBM model trained on text data.

## Contents

- `score.py`: Contains the scoring function used to evaluate a trained model on text data.
- `app.py`: Flask application defining an endpoint for scoring text data using the trained model.
- `test_code.py`: Unit and integration tests for the scoring function and Flask endpoint.
- `coverage.txt`: Coverage report generated using pytest.
- `model`: Directory containing the trained model file (lightgbm_model.pkl).

## Setup and Usage

### Dependencies:
-   Python 3.10
-   Flask
-   Joblib
-   ndas
-   quests
-   Scikit-learn

### Running Tests

To run the tests, execute the following command:
```
python3.10 -m coverage run -a -m pytest -s score.py
python3.10 -m coverage run -a -m pytest -s test_code.py
python3.10 -m coverage run -a -m pytest -s app.py
python3.10 -m coverage report -m > coverage.txt
```

## Test Cases

- **Smoke Test**: Ensures that the scoring function produces output without crashing.
- **Format Test**: Validates the input/output formats/types of the scoring function.
- **Prediction Value Test**: Verifies that the prediction value is either 0 or 1.
- **Propensity Range Test**: Checks if the propensity score is between 0 and 1.
- **Threshold Tests**: Tests if the prediction behaves correctly when the threshold is set to 0 or 1.
- **Spam Text Test**: Checks if an obvious spam text yields a prediction of 1.
- **Non-Spam Text Test**: Verifies that an obvious non-spam text yields a prediction of 0.
- **Flask Integration Test**: Tests the Flask endpoint by sending a POST request with test text data and verifying the response.

## Flask Serving:
* The Flask application is defined in app.py.
* An endpoint /score is provided to receive text data via POST request and return predictions in JSON format.
* Integration tests in test_code.py verify the functionality of the Flask application by launching the server, sending requests, and checking responses.
* Tests ensure proper handling of requests, correct prediction outputs, and response format.

## Coverage Report:
* The coverage report (`coverage.txt`) provides insights into the test coverage of the codebase.
* It shows the percentage of code lines covered by unit tests and integration tests.
* Coverage helps ensure that critical parts of the code are thoroughly tested.

Feel free to reach out with any questions or concerns regarding the project.
