# Predicting Human Age Based on Health Data

## Project Overview

This project involves predicting human age using a synthetic dataset containing various health-related features. It uses multiple data preprocessing steps, feature selection techniques, and machine learning models to achieve a high-performing predictive model. The primary goal is to identify the most important predictors of age and assess various modeling approaches.

---

## Dataset

- **Source**: [Kaggle Dataset - Human Age Prediction (Synthetic)](https://www.kaggle.com/datasets/abdullah0a/human-age-prediction-synthetic-dataset/code)
- **Size**: ~3000 observations
- **Features**:
  - Demographics (e.g., Gender, Education Level)
  - Health Metrics (e.g., Blood Pressure, Cholesterol, Vision Sharpness)
  - Lifestyle Indicators (e.g., Smoking Status, Alcohol Consumption)

---

## Methodology

### 1. Data Preprocessing
- Explored dataset structure and identified missing values.
- Converted `Blood Pressure (s/d)` strings into separate numerical columns (`Systolic` and `Diastolic`).
- Dummified categorical variables for machine learning compatibility.
- Imputed missing values with "None" for categorical columns.

### 2. Feature Engineering
- Addressed multicollinearity by dropping redundant features (e.g., `Height (cm)` and `Weight (kg)` due to their correlation with BMI).
- Detected and analyzed outliers using boxplots and scatterplots but retained them due to their negligible impact on model performance.

### 3. Feature Selection
- Applied Recursive Feature Elimination with Cross-Validation (RFECV).
- Evaluated feature importance using LASSO and Random Forest.
- Selected top predictors: Cholesterol, Blood Glucose Level, Bone Density, Vision Sharpness, Hearing Ability, Cognitive Function, and Blood Pressure metrics.

### 4. Modeling
- Established a baseline model using Linear Regression.
- Compared performance with advanced models:
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)
  - XGBoost Regressor
  - LightGBM Regressor
- **Metrics**: R² Score and RMSE

---

## Results

- **Best Performing Model**: Linear Regression
  - **R² Score**: High
  - **RMSE**: Low
- Despite testing advanced models, the linear regression model provided the most interpretable and efficient results.

---

## Key Insights

- **Bone Density** and **Blood Glucose Level** are significant predictors of age, highlighting the role of metabolic health.
- Vision Sharpness and Cognitive Function are also strong indicators, emphasizing the importance of sensory and mental health.

---

## Recommendations for Future Work

- **Feature Engineering**: Include dimensionality reduction techniques to refine the feature set.
- **Hyperparameter Tuning**: Explore optimal settings for advanced models like Random Forest and Gradient Boosting.
- **Generalizability**: Validate the model on an external dataset to assess robustness.

---

## Project Structure

- **predictingage.ipynb**: Main script for preprocessing, feature selection, and modeling.
- **Train.csv**: Input dataset.
- **Outputs**:
  - Feature importance rankings
  - Model performance metrics
  - Visualization of preprocessing steps and feature relationships

---

## Requirements

- Python Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`
- Visualization Tool: `missingno` for missing data analysis

---

## How to Run

1. Clone the repository.
2. Install the required libraries.
3. Place the `Train.csv` file in the root directory.
4. Execute the script: `python predictingage.ipynb`.

---

This README reflects the project's comprehensive approach to health data analysis and predictive modeling, making it an excellent addition to a data analytics portfolio.
