import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

print("Analytics environment ready!")
# Load Titanic dataset once
df = sns.load_dataset("titanic")

# Save an offline copy for the rest of the project
df.to_csv("analytics/titanic.csv", index=False)

print("\nTitanic dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
# =========================
# 1. DATASET PROFILE
# =========================

print("\n" + "=" * 50)
print("DATASET PROFILE")
print("=" * 50)

print("\n--- INFO ---")
df.info()

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe(include="all"))

print("\n--- MISSING VALUES (%) ---")
missing_percent = df.isnull().mean() * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))
# =========================
# 2. MISSING VALUE HANDLING
# =========================

print("\n" + "=" * 50)
print("MISSING VALUE HANDLING")
print("=" * 50)

# Calculate missing percentage
missing_percent = df.isnull().mean() * 100

print("\nMissing percentages:")
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# Apply threshold-based handling
# <5%  -> drop rows
# 5-30% -> median/mode imputation
# >30% -> drop column

for column in df.columns:
    missing_pct = missing_percent[column]

    if missing_pct == 0:
        continue

    if missing_pct < 5:
        df = df.dropna(subset=[column])
        print(f"{column}: {missing_pct:.2f}% missing -> rows dropped")

    elif missing_pct <= 30:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
            print(
                f"{column}: {missing_pct:.2f}% missing "
                "-> median imputation"
            )
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            print(
                f"{column}: {missing_pct:.2f}% missing "
                "-> mode imputation"
            )

    else:
        df = df.drop(columns=[column])
        print(
            f"{column}: {missing_pct:.2f}% missing "
            "-> column dropped because missingness is very high"
        )

print("\nShape after missing-value handling:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())
# =========================
# 3. UNIVARIATE ANALYSIS
# AGE
# =========================

print("\n" + "=" * 50)
print("AGE ANALYSIS")
print("=" * 50)

# Age Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/age_histogram.png")
plt.show()

# Age Boxplot
plt.figure(figsize=(8, 4))
plt.boxplot(df["age"], vert=False)
plt.title("Age Boxplot")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig("analytics/age_boxplot.png")
plt.show()

# IQR outlier count
Q1_age = df["age"].quantile(0.25)
Q3_age = df["age"].quantile(0.75)
IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = df[
    (df["age"] < lower_age) |
    (df["age"] > upper_age)
]

print(f"Age Q1: {Q1_age:.2f}")
print(f"Age Q3: {Q3_age:.2f}")
print(f"Age IQR: {IQR_age:.2f}")
print(f"Age lower bound: {lower_age:.2f}")
print(f"Age upper bound: {upper_age:.2f}")
print(f"Age outlier count: {len(age_outliers)}")
# =========================
# FARE ANALYSIS
# =========================

print("\n" + "=" * 50)
print("FARE ANALYSIS")
print("=" * 50)

# Fare Histogram
plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=30, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/fare_histogram.png")
plt.show()

# Fare Boxplot
plt.figure(figsize=(8, 4))
plt.boxplot(df["fare"], vert=False)
plt.title("Fare Boxplot")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("analytics/fare_boxplot.png")
plt.show()

# IQR outlier calculation
Q1_fare = df["fare"].quantile(0.25)
Q3_fare = df["fare"].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = df[
    (df["fare"] < lower_fare) |
    (df["fare"] > upper_fare)
]

# Mean, Median, Mode
fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print(f"Fare mean: {fare_mean:.2f}")
print(f"Fare median: {fare_median:.2f}")
print(f"Fare mode: {fare_mode:.2f}")

print(f"Fare Q1: {Q1_fare:.2f}")
print(f"Fare Q3: {Q3_fare:.2f}")
print(f"Fare IQR: {IQR_fare:.2f}")
print(f"Fare lower bound: {lower_fare:.2f}")
print(f"Fare upper bound: {upper_fare:.2f}")
print(f"Fare outlier count: {len(fare_outliers)}")

# Skewness conclusion
if fare_mean > fare_median:
    print("Conclusion: Fare is positively/right skewed.")
elif fare_mean < fare_median:
    print("Conclusion: Fare is negatively/left skewed.")
else:
    print("Conclusion: Fare is approximately symmetric.")
    # =========================
# 4. BIVARIATE ANALYSIS
# SURVIVAL BY SEX
# =========================

print("\n" + "=" * 50)
print("SURVIVAL RATE BY SEX")
print("=" * 50)

male_survival = df.loc[df["sex"] == "male", "survived"].mean()
female_survival = df.loc[df["sex"] == "female", "survived"].mean()

print(f"Male survival rate: {male_survival:.2%}")
print(f"Female survival rate: {female_survival:.2%}")
# =========================
# SURVIVAL RATE BY PCLASS
# =========================

print("\n" + "=" * 50)
print("SURVIVAL RATE BY PCLASS")
print("=" * 50)

for pclass in sorted(df["pclass"].unique()):
    rate = df.loc[df["pclass"] == pclass, "survived"].mean()
    print(f"Class {pclass} survival rate: {rate:.2%}")
    # =========================
# SURVIVAL RATE BY SEX + PCLASS
# =========================

print("\n" + "=" * 50)
print("SURVIVAL RATE BY SEX + PCLASS")
print("=" * 50)

for sex in sorted(df["sex"].unique()):
    for pclass in sorted(df["pclass"].unique()):
        subset = df[
            (df["sex"] == sex) &
            (df["pclass"] == pclass)
        ]

        rate = subset["survived"].mean()

        print(
            f"{sex}, Class {pclass}: "
            f"{rate:.2%}"
        )
        # =========================
# CORRELATION ANALYSIS
# =========================

print("\n" + "=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[corr_columns].corr()

print("\nCorrelation Matrix:")
print(corr_matrix.round(3))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("analytics/correlation_heatmap.png")
plt.show()

# Find top 2 absolute off-diagonal correlations
corr_pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        col1 = corr_columns[i]
        col2 = corr_columns[j]
        value = corr_matrix.loc[col1, col2]

        corr_pairs.append(
            (col1, col2, value, abs(value))
        )

corr_pairs.sort(key=lambda x: x[3], reverse=True)

print("\nTop 2 absolute correlations:")

for col1, col2, value, absolute_value in corr_pairs[:2]:
    print(
        f"{col1} vs {col2}: "
        f"correlation = {value:.3f}"
    )
    # =========================
# 5. MULTIVARIATE ANALYSIS
# CHART 1: SURVIVAL BY SEX + PCLASS
# =========================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()

plt.savefig("analytics/survival_by_sex_pclass.png")
plt.show()
# =========================
# CHART 2: AGE + SURVIVAL
# =========================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()

plt.savefig("analytics/age_by_survival.png")
plt.show()
# =========================
# CHART 3: AGE + FARE + SURVIVAL
# =========================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7
)

plt.title("Age vs Fare by Survival and Sex")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.tight_layout()

plt.savefig("analytics/age_fare_survival.png")
plt.show()
# =========================
# CHART 4: FAMILY SIZE + SURVIVAL
# =========================

df["family_size"] = df["sibsp"] + df["parch"] + 1

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="family_size",
    y="survived"
)

plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()

plt.savefig("analytics/survival_by_family_size.png")
plt.show()
# =========================
# 6. EXPLORATORY STANDARDIZATION
# =========================

print("\n" + "=" * 50)
print("EXPLORATORY STANDARDIZATION")
print("=" * 50)

for column in ["age", "fare"]:
    mean_value = df[column].mean()
    std_value = df[column].std()

    df[f"{column}_z"] = (
        (df[column] - mean_value) / std_value
    )

    print(f"\n{column.upper()} BEFORE STANDARDIZATION")
    print(f"Mean: {df[column].mean():.4f}")
    print(f"Std:  {df[column].std():.4f}")

    print(f"{column.upper()} AFTER STANDARDIZATION")
    print(f"Mean: {df[f'{column}_z'].mean():.4f}")
    print(f"Std:  {df[f'{column}_z'].std():.4f}")
    # =========================
# 7. CLASSIFICATION SETUP
# =========================

print("\n" + "=" * 50)
print("CLASSIFICATION SETUP")
print("=" * 50)

# Remove target and columns that would cause target leakage
X = df.drop(columns=["survived", "alive"])

y = df["survived"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("\nOverall class distribution:")
print(y.value_counts(normalize=True))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True))
# =========================
# 8. PREPROCESSING PIPELINE
# =========================

print("\n" + "=" * 50)
print("PREPROCESSING PIPELINE")
print("=" * 50)

# Drop derived exploratory columns from modeling
drop_columns = ["age_z", "fare_z", "family_size"]

X_train_model = X_train.drop(columns=drop_columns, errors="ignore")
X_test_model = X_test.drop(columns=drop_columns, errors="ignore")

# Identify column types
numeric_features = X_train_model.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train_model.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

print("Numeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# Numeric preprocessing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

print("\nPreprocessing pipeline created successfully!")
# =========================
# 9. CLASSIFICATION MODELS
# =========================

print("\n" + "=" * 50)
print("CLASSIFICATION MODELS")
print("=" * 50)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for model_name, model in models.items():

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Fit ONLY on training data
    model_pipeline.fit(X_train_model, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test_model)

    # Probability for ROC-AUC
    y_prob = model_pipeline.predict_proba(X_test_model)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(f"\n{model_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4))
# =========================
# 10. CONFUSION MATRICES
# =========================

print("\n" + "=" * 50)
print("CONFUSION MATRICES")
print("=" * 50)

for model_name, model in models.items():

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    model_pipeline.fit(X_train_model, y_train)

    y_pred = model_pipeline.predict(X_test_model)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name}")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(f"analytics/{filename}")
    plt.show()
    # =========================
# 11. DECISION TREE VISUALIZATION
# =========================

print("\n" + "=" * 50)
print("DECISION TREE VISUALIZATION")
print("=" * 50)

# Fit preprocessing separately so we can get feature names
X_train_transformed = preprocessor.fit_transform(X_train_model)

# Train a decision tree on transformed training data
tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

tree_model.fit(X_train_transformed, y_train)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")
plt.tight_layout()

plt.savefig(
    "analytics/decision_tree.png",
    dpi=150
)

plt.show()

print("Decision tree plot saved successfully!")
# =========================
# 12. CLASS IMBALANCE
# BASELINE VS CLASS_WEIGHT
# =========================

print("\n" + "=" * 50)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 50)

print("\nClass distribution:")
print(y.value_counts())
print("\nClass proportions:")
print(y.value_counts(normalize=True))

imbalance_results = []

# Baseline Random Forest
baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

baseline_pipeline.fit(X_train_model, y_train)

baseline_pred = baseline_pipeline.predict(X_test_model)

imbalance_results.append({
    "Method": "Baseline",
    "Precision": precision_score(y_test, baseline_pred),
    "Recall": recall_score(y_test, baseline_pred),
    "F1": f1_score(y_test, baseline_pred)
})

# Balanced Random Forest
balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

balanced_pipeline.fit(X_train_model, y_train)

balanced_pred = balanced_pipeline.predict(X_test_model)

imbalance_results.append({
    "Method": "Class Weight Balanced",
    "Precision": precision_score(y_test, balanced_pred),
    "Recall": recall_score(y_test, balanced_pred),
    "F1": f1_score(y_test, balanced_pred)
})

imbalance_results_df = pd.DataFrame(imbalance_results)

print("\nBaseline vs Class Weight:")
print(imbalance_results_df.round(4))
# SMOTE - applied only to training data
smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]
)

smote_pipeline.fit(X_train_model, y_train)

smote_pred = smote_pipeline.predict(X_test_model)

imbalance_results.append({
    "Method": "SMOTE",
    "Precision": precision_score(y_test, smote_pred),
    "Recall": recall_score(y_test, smote_pred),
    "F1": f1_score(y_test, smote_pred)
})

imbalance_results_df = pd.DataFrame(imbalance_results)

print("\nBaseline vs Class Weight vs SMOTE:")
print(imbalance_results_df.round(4))
# =========================
# 13. RANDOM FOREST GRID SEARCH
# =========================

print("\n" + "=" * 50)
print("RANDOM FOREST GRID SEARCH")
print("=" * 50)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            oob_score=True,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train_model, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1:")
print(grid_search.best_score_)

best_rf = grid_search.best_estimator_

print("\nOOB Score:")
print(best_rf.named_steps["model"].oob_score_)
# =========================
# 14. MULTIVARIATE LINEAR REGRESSION
# Predict Fare
# =========================

print("\n" + "=" * 50)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 50)

# Remove target and derived columns
regression_drop = [
    "fare",
    "fare_z",
    "alive",
    "family_size"
]

X_reg = df.drop(columns=regression_drop, errors="ignore")
y_reg = df["fare"]

# Same random_state for reproducibility
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

# Identify feature types
reg_numeric_features = X_reg_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

reg_categorical_features = X_reg_train.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

# Numeric preprocessing
reg_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing
reg_categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", reg_numeric_pipeline, reg_numeric_features),
        ("categorical", reg_categorical_pipeline, reg_categorical_features)
    ]
)

# Linear Regression pipeline
regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ]
)

# Train
regression_pipeline.fit(X_reg_train, y_reg_train)

# Predict
y_reg_pred = regression_pipeline.predict(X_reg_test)

# Metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

# Adjusted R²
X_reg_test_transformed = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_reg_test)

n = len(y_reg_test)
p = X_reg_test_transformed.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nRegression Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")

# Residuals
residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_reg_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Prediction")
plt.tight_layout()
plt.savefig("analytics/fare_residual_plot.png", dpi=150)
plt.show()

print("\nResidual plot saved successfully!")
# =========================
# 15. SAVE AND RELOAD BEST MODEL
# =========================

print("\n" + "=" * 50)
print("SAVING BEST MODEL")
print("=" * 50)

model_path = "analytics/best_model.joblib"

# Save the complete fitted pipeline
joblib.dump(best_rf, model_path)

print(f"\nModel saved to: {model_path}")

# Reload model
loaded_model = joblib.load(model_path)

print("Model reloaded successfully!")

# Predict using raw test data
sample_prediction = loaded_model.predict(X_test_model.head(5))

print("\nSample predictions after reload:")
print(sample_prediction)
# =========================
# 16. FINAL MODEL COMPARISON
# =========================

print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

# Evaluate tuned Random Forest
best_rf_pred = best_rf.predict(X_test_model)
best_rf_prob = best_rf.predict_proba(X_test_model)[:, 1]

best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_precision = precision_score(y_test, best_rf_pred)
best_rf_recall = recall_score(y_test, best_rf_pred)
best_rf_f1 = f1_score(y_test, best_rf_pred)
best_rf_auc = roc_auc_score(y_test, best_rf_prob)

# Recalculate metrics for Logistic Regression
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]
)

logistic_pipeline.fit(X_train_model, y_train)
logistic_pred = logistic_pipeline.predict(X_test_model)
logistic_prob = logistic_pipeline.predict_proba(X_test_model)[:, 1]

# Recalculate metrics for Decision Tree
tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)

tree_pipeline.fit(X_train_model, y_train)
tree_pred = tree_pipeline.predict(X_test_model)
tree_prob = tree_pipeline.predict_proba(X_test_model)[:, 1]

# Final classification table
final_classification_results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, logistic_pred),
        "Precision": precision_score(y_test, logistic_pred),
        "Recall": recall_score(y_test, logistic_pred),
        "F1": f1_score(y_test, logistic_pred),
        "ROC-AUC": roc_auc_score(y_test, logistic_prob)
    },
    {
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(y_test, tree_pred),
        "Precision": precision_score(y_test, tree_pred),
        "Recall": recall_score(y_test, tree_pred),
        "F1": f1_score(y_test, tree_pred),
        "ROC-AUC": roc_auc_score(y_test, tree_prob)
    },
    {
        "Model": "Tuned Random Forest",
        "Accuracy": best_rf_accuracy,
        "Precision": best_rf_precision,
        "Recall": best_rf_recall,
        "F1": best_rf_f1,
        "ROC-AUC": best_rf_auc
    }
])

print("\nClassification Model Comparison:")
print(final_classification_results.round(4))

print("\nRegression Model:")
print(
    pd.DataFrame([{
        "Model": "Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adjusted_r2
    }]).round(4)
)

# Save comparison tables
final_classification_results.to_csv(
    "analytics/classification_model_comparison.csv",
    index=False
)

pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted R2": adjusted_r2
}]).to_csv(
    "analytics/regression_model_comparison.csv",
    index=False
)

print("\nComparison tables saved successfully!")