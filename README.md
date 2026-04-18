# Rainfall Prediction using Machine Learning (End-to-End Pipeline)

## Project Overview

This project builds an end-to-end machine learning pipeline to predict whether it will rain the next day using historical weather data from Australia (Taken from Kaggle).

The workflow includes: 
data preprocessing, 
feature engineering, 
model training, 
hyperparameter tuning, and 
evaluation using industry-standard techniques such as cross-validation and grid search.

The goal is to develop a reliable classification model capable of identifying rainfall patterns based on meteorological features.

## Dataset

The dataset contains daily weather observations from multiple locations across Australia, including features such as:

- Temperature (MinTemp, MaxTemp)
- Rainfall
- Humidity levels
- Wind speed and direction
- Atmospheric pressure
- Cloud coverage
- Sunshine hours

Target variable:
- `RainTomorrow` (Yes/No)

## What did i do??

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical variables using One-Hot Encoding
- Scaled numerical features where necessary

### 2. Pipeline Construction
- Built a Scikit-learn pipeline to streamline preprocessing and modeling
- Used ColumnTransformer to process categorical and numerical features separately

### 3. Model Training
- Trained a Random Forest Classifier as the baseline model
- Switched to Logistic Regression for comparison

### 4. Hyperparameter Tuning
- Applied GridSearchCV with Stratified K-Fold Cross Validation
- Optimized model parameters to improve performance

### 5. Model Evaluation
- Evaluated using:
  - Accuracy
  - Precision
  - Recall (True Positive Rate)
  - Confusion Matrix
- Analyzed performance trade-offs between models

## End results?

- Best cross-validation accuracy: ~85%
- Model demonstrates strong performance in predicting no-rain cases
- Moderate recall for rain detection (~50%), indicating room for improvement in identifying positive rainfall events

Confusion Matrix analysis shows:
- High True Negatives (correctly predicting no rain)
- Lower True Positives (missing some rain events)

## Interpretations and learnings -
- Class imbalance significantly impacts model performance
- Accuracy alone is not sufficient; recall is critical for this problem
- Logistic Regression provides interpretability, while Random Forest captures non-linear patterns
- Feature importance reveals humidity, pressure, and cloud coverage as strong predictors



