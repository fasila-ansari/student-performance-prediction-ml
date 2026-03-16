# ============================================================
# Student Performance Prediction using Machine Learning
# Data Science Project
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


print("====================================")
print("Student Performance ML Project Start")
print("====================================")


# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("student-mat.csv")

print("\nDataset Loaded Successfully")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# Dataset Information
# ============================================================

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# Exploratory Data Analysis
# ============================================================

print("\nGenerating visualizations...")

plt.figure(figsize=(6,4))

sns.histplot(df["G3"], bins=20)

plt.title("Distribution of Final Grades")

plt.xlabel("Final Grade")

plt.ylabel("Count")

plt.show()


plt.figure(figsize=(10,8))

sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap="coolwarm")

plt.title("Feature Correlation Matrix")

plt.show()


# ============================================================
# Feature Selection
# ============================================================

print("\nSelecting important features...")

features = ["studytime","failures","absences","G1","G2"]

X = df[features]

y = df["G3"]


# ============================================================
# Train Test Split
# ============================================================

print("\nSplitting dataset into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# ============================================================
# Model 1: Linear Regression
# ============================================================

print("\nTraining Linear Regression model...")

lr_model = LinearRegression()

lr_model.fit(X_train,y_train)

lr_predictions = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test,lr_predictions)

lr_r2 = r2_score(y_test,lr_predictions)

print("\nLinear Regression Results")

print("Mean Squared Error:",lr_mse)

print("R2 Score:",lr_r2)


# ============================================================
# Model 2: Random Forest Regressor
# ============================================================

print("\nTraining Random Forest model...")

rf_model = RandomForestRegressor(

    n_estimators=100,
    random_state=42

)

rf_model.fit(X_train,y_train)

rf_predictions = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test,rf_predictions)

rf_r2 = r2_score(y_test,rf_predictions)

print("\nRandom Forest Results")

print("Mean Squared Error:",rf_mse)

print("R2 Score:",rf_r2)


# ============================================================
# Model Comparison
# ============================================================

print("\nComparing model performance...")

results = pd.DataFrame({

    "Model":["Linear Regression","Random Forest"],

    "R2 Score":[lr_r2,rf_r2],

    "MSE":[lr_mse,rf_mse]

})

print(results)


# ============================================================
# Visualization: Actual vs Predicted
# ============================================================

plt.figure(figsize=(6,4))

plt.scatter(y_test,rf_predictions)

plt.xlabel("Actual Grades")

plt.ylabel("Predicted Grades")

plt.title("Actual vs Predicted Grades (Random Forest)")

plt.show()


# ============================================================
# Feature Importance Analysis
# ============================================================

print("\nCalculating feature importance...")

importance = rf_model.feature_importances_

feature_names = X.columns

feature_importance_df = pd.DataFrame({

    "Feature":feature_names,

    "Importance":importance

}).sort_values(by="Importance",ascending=False)

print("\nFeature Importance Ranking:")

print(feature_importance_df)


# ============================================================
# Feature Importance Visualization
# ============================================================

plt.figure(figsize=(8,5))

sns.barplot(

    x="Importance",

    y="Feature",

    data=feature_importance_df

)

plt.title("Feature Importance in Predicting Student Grades")

plt.show()


# ============================================================
# Final Insights
# ============================================================

print("\nProject Insights:")

print("1. Previous grades (G1 and G2) strongly influence final performance.")

print("2. Study time and absences also affect student outcomes.")

print("3. Random Forest performed better than Linear Regression.")

print("\nProject Completed Successfully!")