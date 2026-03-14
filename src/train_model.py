import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from preprocess import load_data, prepare_features

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "crop_prices.csv"

df = load_data(data_path)
X, y = prepare_features(df)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

best_model = None
best_score = float("-inf")
best_model_name = ""

print("Model Performance (5-fold Cross Validation):\n")
for name, model in models.items():

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    mean_score = np.mean(scores)

    print(f"{name} Mean R2 Score: {mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_model_name = name


print("\nBest Model:", best_model_name)
best_model.fit(X, y)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "price_model.pkl"

joblib.dump(best_model, model_path)
print("\nModel saved successfully to models/price_model.pkl")
