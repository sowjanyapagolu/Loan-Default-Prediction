# Loan Default Prediction System

## Objective
Predict loan-default probability using borrower financial data.

## Technologies
- Python
- XGBoost
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Evidently AI
- Docker
- GitHub Actions

## Features
- Dataset generation
- Machine learning model training
- Model evaluation
- Experiment tracking
- Dataset versioning
- Prediction API
- Data drift monitoring
- Docker deployment
- Automated testing

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset:

```bash
python src/generate_data.py
```

Train the model:

```bash
python src/train.py
```

Start the API:

```bash
python -m uvicorn app:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

## Disclaimer
This educational project uses synthetic data and is not intended for real lending decisions.