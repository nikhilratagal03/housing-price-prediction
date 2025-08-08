# Housing Price Prediction

This project predicts house prices using the King County housing dataset. The core model is an XGBoost regressor trained on features like bedrooms, bathrooms, square footage, location, house age, and renovation status.

---

## Project Overview

- The dataset is cleaned and preprocessed, with features engineered to improve prediction (e.g., house age and renovation flag).
- The target variable (house price) is log-transformed to stabilize variance and improve model accuracy.
- An XGBoost regression model is trained with tuned hyperparameters (`n_estimators=100`, `learning_rate=0.1`, `max_depth=6`).
- Model evaluation uses Root Mean Squared Error (RMSE) and R² score on the original price scale after inverse log transformation.
- A Streamlit app is developed to allow users to input house features and get instant price predictions.
- The trained model is saved for fast inference during deployment.

---

## Dataset

The project uses the [King County House Sales dataset](https://www.kaggle.com/harlfoxem/housesalesprediction), which includes house sale prices and features such as bedrooms, bathrooms, square footage, zipcode, latitude, longitude, year built, and renovation info.

---

## Model Details

- Model: XGBoost Regressor  
- Hyperparameters: 100 trees, learning rate of 0.1, max depth of 6  
- Target: Log-transformed house price  
- Evaluation metrics: RMSE and R² Score on original price scale

---

## Why XGBoost?

XGBoost is used instead of Random Forest due to its boosting technique that iteratively corrects errors, regularization to prevent overfitting, and high computational efficiency — leading to better predictive accuracy.

---

## Deployment
The model is deployed via a Streamlit web app, enabling easy interaction for users to predict house prices based on input features.

## Acknowledgments

Thank you for your interest in this project. I sincerely hope this work helps you understand housing price prediction and inspires you to build your own machine learning applications.

Happy coding!

— Nikhil Ratagal

— Nikhil Ratagal

