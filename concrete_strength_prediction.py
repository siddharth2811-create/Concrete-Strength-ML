# ============================================================
# MACHINE LEARNING-BASED PREDICTION OF CONCRETE COMPRESSIVE
# STRENGTH
# ============================================================
#
# Project:
# Predicting concrete compressive strength using machine
# learning regression models.
#
# Dataset:
# UCI Concrete Compressive Strength Dataset
#
# Models:
# 1. Linear Regression
# 2. Decision Tree Regression
# 3. Random Forest Regression
#
# Evaluation Metrics:
# MAE, RMSE, R²
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("CONCRETE COMPRESSIVE STRENGTH PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

# Fetch Concrete Compressive Strength dataset
concrete = fetch_ucirepo(id=165)

# Separate features and target
X = concrete.data.features
y = concrete.data.targets

print("Dataset loaded successfully.")


# ============================================================
# 3. PREPARE DATASET
# ============================================================

# Rename columns for easier handling
X.columns = [
    "Cement",
    "Blast_Furnace_Slag",
    "Fly_Ash",
    "Water",
    "Superplasticizer",
    "Coarse_Aggregate",
    "Fine_Aggregate",
    "Age"
]

y.columns = ["Compressive_Strength"]


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeatures:")
for feature in X.columns:
    print("-", feature)

print("\nTarget:")
print("- Compressive_Strength")

print("\nMissing values:")
print(X.isnull().sum())

print("\nDataset statistics:")
print(X.describe())


# ============================================================
# 5. COMBINE FEATURES AND TARGET
# ============================================================

data = pd.concat([X, y], axis=1)


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Distribution of compressive strength
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    y["Compressive_Strength"],
    bins=30,
    edgecolor="black"
)

plt.xlabel("Compressive Strength (MPa)")
plt.ylabel("Number of Samples")
plt.title("Distribution of Concrete Compressive Strength")

plt.tight_layout()

plt.savefig(
    "compressive_strength_distribution.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------

correlation = data.corr()

print("\nCorrelation with Compressive Strength:")
print(
    correlation["Compressive_Strength"]
    .sort_values(ascending=False)
)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

linear_model = LinearRegression()

# Train model
linear_model.fit(
    X_train,
    y_train
)

# Predictions
linear_predictions = linear_model.predict(X_test)

# Evaluation
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("MAE :", round(linear_mae, 3))
print("RMSE:", round(linear_rmse, 3))
print("R²  :", round(linear_r2, 3))


# ============================================================
# 9. DECISION TREE REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE REGRESSION")
print("=" * 60)

tree_model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

# Train model
tree_model.fit(
    X_train,
    y_train
)

# Predictions
tree_predictions = tree_model.predict(X_test)

# Evaluation
tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_predictions
    )
)

tree_r2 = r2_score(
    y_test,
    tree_predictions
)

print("MAE :", round(tree_mae, 3))
print("RMSE:", round(tree_rmse, 3))
print("R²  :", round(tree_r2, 3))


# ============================================================
# 10. RANDOM FOREST REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION")
print("=" * 60)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(
    X_train,
    y_train.values.ravel()
)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("MAE :", round(rf_mae, 3))
print("RMSE:", round(rf_rmse, 3))
print("R²  :", round(rf_r2, 3))


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        tree_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        tree_rmse,
        rf_rmse
    ],

    "R2": [
        linear_r2,
        tree_r2,
        rf_r2
    ]
})

print("\n")
print(results.to_string(index=False))


# ============================================================
# 12. MODEL COMPARISON PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["R2"]
)

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Comparison of Model Performance")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test["Compressive_Strength"],
    rf_predictions,
    alpha=0.7
)

# Perfect prediction line
minimum = min(
    y_test["Compressive_Strength"].min(),
    rf_predictions.min()
)

maximum = max(
    y_test["Compressive_Strength"].max(),
    rf_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Compressive Strength (MPa)")
plt.ylabel("Predicted Compressive Strength (MPa)")

plt.title(
    "Actual vs Predicted Concrete Compressive Strength"
)

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print(importance.to_string(index=False))


# ============================================================
# 15. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. PREDICTION FOR A NEW CONCRETE MIX
# ============================================================

print("\n" + "=" * 60)
print("NEW CONCRETE MIX PREDICTION")
print("=" * 60)

# Example concrete mix
new_concrete = pd.DataFrame({
    "Cement": [350],
    "Blast_Furnace_Slag": [100],
    "Fly_Ash": [50],
    "Water": [180],
    "Superplasticizer": [8],
    "Coarse_Aggregate": [1000],
    "Fine_Aggregate": [700],
    "Age": [28]
})

# Predict compressive strength
prediction = rf_model.predict(
    new_concrete
)

print(
    "\nPredicted Compressive Strength:",
    round(prediction[0], 2),
    "MPa"
)


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(
    rf_model,
    "concrete_strength_random_forest.pkl"
)

print(
    "Random Forest model saved as "
    "'concrete_strength_random_forest.pkl'"
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results.to_csv(
    "model_comparison_results.csv",
    index=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nResults saved successfully.")


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("\nModels evaluated:")
print("1. Linear Regression")
print("2. Decision Tree Regression")
print("3. Random Forest Regression")

print("\nEvaluation metrics:")
print("1. MAE")
print("2. RMSE")
print("3. R²")

print("\nGenerated files:")
print("- compressive_strength_distribution.png")
print("- correlation_heatmap.png")
print("- model_comparison.png")
print("- actual_vs_predicted.png")
print("- feature_importance.png")
print("- model_comparison_results.csv")
print("- feature_importance.csv")
print("- concrete_strength_random_forest.pkl")

print("\nProject completed successfully!")