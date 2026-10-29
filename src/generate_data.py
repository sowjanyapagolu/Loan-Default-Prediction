from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
n = 1000

income = np.random.randint(20000, 150000, n)
loan_amount = np.random.randint(5000, 100000, n)
credit_score = np.random.randint(300, 851, n)
existing_debt = np.random.randint(0, 50000, n)
previous_defaults = np.random.randint(0, 3, n)

risk = (
    (loan_amount / income) * 0.8
    + (700 - credit_score) / 250
    + existing_debt / 100000
    + previous_defaults * 0.5
)

probability = 1 / (1 + np.exp(-risk))
default = np.random.binomial(1, probability)

df = pd.DataFrame({
    "income": income,
    "loan_amount": loan_amount,
    "credit_score": credit_score,
    "existing_debt": existing_debt,
    "previous_defaults": previous_defaults,
    "default": default
})

df.to_csv(DATA_DIR / "loan_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())
print("\nDefault distribution:")
print(df["default"].value_counts())