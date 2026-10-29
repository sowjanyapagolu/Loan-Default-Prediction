
from pathlib import Path

import joblib
import pandas as pd
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"

app = FastAPI(title="Loan Default Prediction System")

model = joblib.load(MODEL_PATH)


class LoanApplicant(BaseModel):
    income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    existing_debt: float = Field(ge=0)
    previous_defaults: int = Field(ge=0, le=2)


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(data: LoanApplicant):
    try:
        applicant = pd.DataFrame([data.model_dump()])

        # Match the feature order used during model training
        if hasattr(model, "feature_names_in_"):
            applicant = applicant[list(model.feature_names_in_)]

        probability = float(model.predict_proba(applicant)[0][1])

        if probability < 0.30:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Medium"
        else:
            risk_level = "High"

        predicted_default = int(probability >= 0.5)

        return {
            "default_probability": round(probability, 4),
            "risk_level": risk_level,
            "predicted_default": predicted_default,
            "prediction_message": (
                "Loan Default Predicted"
                if predicted_default == 1
                else "No Loan Default Predicted"
            )
        }

    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
