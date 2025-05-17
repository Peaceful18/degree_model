import joblib
import pandas as pd
import numpy as np

# === Завантаження моделі та PCA ===
model = joblib.load("best_model_lightgbm.pkl")
pca = joblib.load("pca_model_lightgbm.pkl")

# === Список файлів для прогнозу ===
file_paths = [
    "SpectrumFile_006.csv",
    "SpectrumFile_007.csv",
    "SpectrumFile_008.csv",
    "SpectrumFile_009.csv",
    "SpectrumFile_010.csv",
    "SpectrumFile_011.csv",
]

# === Оцінка файлів ===
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path, sep=";", header=0)
        spectrum = df["Counts"].values

        # Перетворення спектра через PCA
        spectrum_transformed = pca.transform([spectrum])

        # Прогноз
        predicted_day = model.predict(spectrum_transformed)[0]
        print(f"Файл: {file_path} -> Прогнозований вік молока: {predicted_day:.2f} днів")

    except Exception as e:
        print(f"Помилка з файлом {file_path}: {e}")
