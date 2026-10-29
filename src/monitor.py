
from pathlib import Path
import pandas as pd
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Loan-Default-Prediction")

# Load loan dataset
df = pd.read_csv(ROOT / "data" / "loan_data.csv")

# Separate features
features = df.drop(columns=["default"])

# Split reference and current data
split_index = int(len(features) * 0.8)

reference_data = features.iloc[:split_index].copy()
current_data = features.iloc[split_index:].copy()

# Generate drift report
report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report
report_dir = ROOT / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report_path = report_dir / "data_drift_report.html"
report.save_html(str(report_path))

# Log report to MLflow
with mlflow.start_run(run_name="Data-Drift-Monitoring"):
    mlflow.log_param("reference_rows", len(reference_data))
    mlflow.log_param("current_rows", len(current_data))
    mlflow.log_param("feature_count", len(features.columns))

    mlflow.log_artifact(
        str(report_path),
        artifact_path="drift_report"
    )

    mlflow.set_tag("task", "data_drift_monitoring")

print("Data drift report generated successfully!")
print(f"Location: {report_path}")
print("Monitoring results logged to MLflow!")