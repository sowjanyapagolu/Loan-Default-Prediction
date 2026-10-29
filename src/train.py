
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run: python src/generate_data.py"
    )

# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["default"])
y = df["default"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

# Configure MLflow to use the project database
DATABASE_PATH = (ROOT / "mlflow.db").as_posix()
mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")
mlflow.set_experiment("Loan-Default-Prediction")

# Train and track experiment
with mlflow.start_run(run_name="XGBoost-Training"):

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    # Log parameters
    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "test_size": 0.2,
        "random_state": 42,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "dataset": DATA_PATH.name
    })

    # Log evaluation metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc
    })

    # Log model artifact
    mlflow.xgboost.log_model(
        model,
        artifact_path="model"
    )

    # Save model for FastAPI
    model_path = MODEL_DIR / "loan_model.pkl"
    joblib.dump(model, model_path)

    # Log the saved model file as an additional artifact
    mlflow.log_artifact(str(model_path), artifact_path="model_files")

    print("\nModel trained successfully!")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Model saved:", model_path)
    print("MLflow database:", DATABASE_PATH)
