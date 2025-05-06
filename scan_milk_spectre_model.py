import joblib
import pandas as pd
import numpy as np

# Завантаження моделі та PCA
model = joblib.load("best_model_lightgbm.pkl")
pca = joblib.load("pca_model_lightgbm.pkl")

# Зчитування нового спектра
file_path = "day17.csv"
df = pd.read_csv(file_path, sep=";", header=0)
spectrum = df["Counts"].values

# Перетворення спектра через PCA
spectrum_transformed = pca.transform([spectrum])

# Прогноз
predicted_day = model.predict(spectrum_transformed)[0]
print(f"Прогнозований вік молока: {predicted_day:.2f} днів")
