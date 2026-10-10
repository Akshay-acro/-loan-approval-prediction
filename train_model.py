import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading loan dataset...")

data = pd.read_csv("loan_data.csv")

if "Loan_Status" not in data.columns:
    raise ValueError("Dataset must contain Loan_Status column.")

data = data.dropna(subset=["Loan_Status"])

feature_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Employment_Status",
    "Property_Area"
]

missing_columns = [
    column for column in feature_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

X = data[feature_columns].copy()

y = data["Loan_Status"].astype(str).str.strip().str.upper()

valid_rows = y.isin(["Y", "N"])
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Dataset must contain at least two examples of each class."
    )

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

categorical_features = [
    "Employment_Status",
    "Property_Area"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training the model...")

model.fit(X_train, y_train)

print("Evaluating the model...")

y_pred = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=["N", "Y"],
    target_names=["Rejected", "Approved"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["N", "Y"]))

joblib.dump(model, "loan_model.pkl")

print("\nTrained model saved as loan_model.pkl")
print("Model training completed successfully!")