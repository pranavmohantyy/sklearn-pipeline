# sklearn-pipeline

## Overview
This project implements a production-ready machine learning workflow using the Titanic dataset. It includes feature engineering, model selection, and hyperparameter tuning.

## Methodology
1. **Data Loading**: The dataset is loaded from a URL.
2. **Feature Engineering**: Key features are extracted, missing values are handled, and categorical variables are transformed.
3. **Model Selection**: Various models are evaluated using cross-validation and grid search for hyperparameter tuning.
4. **Predictions**: A FastAPI endpoint is set up for making predictions based on user input.

## Results
The model achieves a satisfactory accuracy score on the test set, with several features contributing significantly to the prediction of survival rates.

## Retraining the Model
To retrain the model:
1. Ensure you have the required packages installed (e.g., scikit-learn, pandas, FastAPI).
2. Update the data if necessary.
3. Run the model_selection.py script to evaluate and select the best model.
4. Save the trained model using pickle.
5. Start the FastAPI server to serve predictions.

## Running the API
Run the following command to start the FastAPI server:
```
uvicorn src.prediction_endpoint:app --reload
```

## Testing
You can run tests for feature engineering using pytest:
```
pytest tests/test_feature_engineering.py
```
