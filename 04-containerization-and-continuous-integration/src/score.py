import pandas as pd
from sklearn.base import BaseEstimator

def score(text: str, model: BaseEstimator, threshold: float):
    # Preprocess the input text
    df = pd.DataFrame({"text": [text]})
    X = model.named_steps["tfidf"].transform(df["text"])

    # Get the model prediction probabilities
    probabilities = model.named_steps["clf"].predict_proba(X)

    # Get the propensity score for the positive class (spam)
    propensity = probabilities[0, 1]

    # Make the final prediction based on the threshold
    prediction = bool(propensity >= threshold)

    return prediction, propensity