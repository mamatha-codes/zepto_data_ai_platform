# Analytics Pipeline

This module performs exploratory data analysis, preprocessing, classification, regression, model evaluation, hyperparameter tuning, and model persistence using the Titanic dataset.

## Dataset

The Titanic dataset is loaded using Seaborn and immediately saved locally as:

`titanic.csv`

The locally saved CSV is used for subsequent analysis.

## Exploratory Data Analysis

The analysis includes:

- Dataset shape and structure
- Statistical summary
- Missing-value percentage analysis
- Missing-value handling
- Age histogram and boxplot
- Fare histogram and boxplot
- Fare mean, median, mode, and skewness analysis
- Survival analysis by sex
- Survival analysis by passenger class
- Correlation heatmap
- Multivariate visualizations

Generated visualizations include:

- `age_histogram.png`
- `age_boxplot.png`
- `fare_histogram.png`
- `fare_boxplot.png`
- `correlation_heatmap.png`
- `survival_by_sex_pclass.png`
- `age_by_survival.png`
- `age_fare_survival.png`
- `survival_by_family_size.png`

## Preprocessing

The pipeline includes:

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Train/test split with `random_state=42`
- Target leakage prevention by removing `alive`

Derived exploratory variables such as `age_z`, `fare_z`, and `family_size` are not used as model inputs.

## Classification

Three classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Generated outputs include:

- `logistic_regression_confusion_matrix.png`
- `decision_tree_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `decision_tree.png`

## Class Imbalance

Class imbalance was evaluated using:

- Baseline model
- Class-weight balanced model
- SMOTE

SMOTE is applied only to the training data through the modelling pipeline to avoid test-data leakage.

## Hyperparameter Tuning

Random Forest was tuned using `GridSearchCV`.

The search includes:

- `n_estimators`
- `max_depth`
- `max_features`

Five-fold cross-validation was used.

Out-of-bag evaluation was also enabled using `oob_score=True`.

## Regression

A multivariate Linear Regression model predicts passenger fare using the remaining available features.

Metrics:

- MAE
- RMSE
- R²
- Adjusted R²

A residual plot is generated to inspect the relationship between predictions and residuals.

Generated output:

- `fare_residual_plot.png`

## Model Persistence

The tuned Random Forest pipeline is saved using Joblib:

`best_model.joblib`

The saved pipeline is reloaded and tested on sample raw test data to verify that the complete preprocessing and prediction workflow works correctly.

## Comparison Tables

Generated CSV files:

- `classification_model_comparison.csv`
- `regression_model_comparison.csv`

These contain the final evaluation metrics for the classification and regression models.

## Main Files

- `analytics.py` — complete analytics and modelling pipeline
- `titanic.csv` — locally saved Titanic dataset
- `requirements.txt` — Python dependencies
- `best_model.joblib` — saved tuned Random Forest pipeline