import joblib
import numpy as np

model = joblib.load("../models/price_model.pkl")

def predict_price(min_price, max_price):

    features = np.array([[min_price, max_price]])

    prediction = model.predict(features)

    return prediction[0]
