import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/heart.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.columns)

# Handle target column
if "target" in df.columns:
    target_column = "target"
elif "Target" in df.columns:
    target_column = "Target"
elif "num" in df.columns:
    target_column = "num"
else:
    raise ValueError("Target column not found.")

# If target contains values greater than 1,
# convert them into binary classification
if df[target_column].nunique() > 2:
    df[target_column] = (df[target_column] > 0).astype(int)

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert everything to numeric where possible
X = X.apply(pd.to_numeric, errors="coerce")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Logistic Regression
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Random Forest
random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# Train models
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

# Select model
if rf_accuracy >= logistic_accuracy:
    final_model = random_forest_model
    model_name = "Random Forest"
else:
    final_model = logistic_model
    model_name = "Logistic Regression"

# Save
joblib.dump(
    final_model,
    "models/heart_model.pkl"
)

print("\nSelected model:", model_name)
print("Heart disease model saved successfully!")