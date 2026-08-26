import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv(
    "data/diabetes.csv",
    header=None,
    names=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
    ]
)

print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# In PIMA data, zero can represent missing/invalid measurements
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in zero_as_missing:
    if column in X.columns:
        X[column] = X[column].replace(0, np.nan)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Logistic Regression pipeline
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Random Forest pipeline
random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train
logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test)
rf_pred = random_forest_model.predict(X_test)

# Evaluation
logistic_accuracy = accuracy_score(y_test, logistic_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nLogistic Regression Accuracy:",
      logistic_accuracy)

print("\nRandom Forest Accuracy:",
      rf_accuracy)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Select better model based on test accuracy
if rf_accuracy >= logistic_accuracy:
    final_model = random_forest_model
    model_name = "Random Forest"
else:
    final_model = logistic_model
    model_name = "Logistic Regression"

# Save model
joblib.dump(
    final_model,
    "models/diabetes_model.pkl"
)

print("\nSelected model:", model_name)
print("Model saved successfully!")