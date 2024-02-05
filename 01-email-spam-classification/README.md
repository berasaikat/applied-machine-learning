# Email Spam Classification Repository

This repository contains code and data for a machine learning project focused on classifying emails as spam or not spam (ham). The project is organized into two Jupyter Notebooks: `prepare.ipynb` for data loading and pre-processing, and `train.ipynb` for training and evaluating machine learning models. The dataset used is stored in the `data` folder.

## Files

### Data Files
- **emails.csv**: The original dataset containing email text and labels.
- **test_X.csv**: Test set features (email text).
- **test_y.csv**: Test set labels (spam or not spam).
- **train_X.csv**: Training set features.
- **train_y.csv**: Training set labels.
- **val_X.csv**: Validation set features.
- **val_y.csv**: Validation set labels.

### Source Code
- **prepare.ipynb**: Jupyter Notebook for loading and visualizing the dataset, as well as pre-processing the text data.
- **train.ipynb**: Jupyter Notebook for loading pre-processed data, training machine learning models (XGBoost, Logistic Regression, Multinomial Naive Bayes), and evaluating their performance.

## Usage

1. Open `prepare.ipynb` to load the dataset, visualize data distribution, and pre-process the text data.
2. Run `train.ipynb` to train machine learning models (XGBoost, Logistic Regression, Multinomial Naive Bayes) and evaluate their performance on the validation set.
3. The final chosen model (XGBoost) is then retrained on the entire training set and evaluated on the test set.

## Dependencies

Ensure you have the following Python libraries installed to run the code:

- pandas
- matplotlib
- seaborn
- scikit-learn
- nltk
- xgboost

## Dataset

The original dataset (`emails.csv`) is used for training and testing. It contains email text and corresponding labels (spam or not spam). The pre-processed datasets (`train_X.csv`, `train_y.csv`, `val_X.csv`, `val_y.csv`, `test_X.csv`, `test_y.csv`) are stored in the `data` folder.

## Results

The XGBoost model achieved a test accuracy of 99.48%. The results are highly satisfactory, and there was no need for further hyperparameter tuning.

Feel free to explore and adapt the code for your own email spam classification projects!
