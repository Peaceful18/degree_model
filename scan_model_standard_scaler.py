import joblib
import pandas as pd
import numpy as np
import os


def predict_milk_age(file_path, model_name="lightgbm_model_scaler"):
    try:
        # Перевіряємо, чи існує файл
        if not os.path.exists(file_path):
            print(f"[!] Помилка: Файл {file_path} не знайдено")
            return None

        # Завантаження моделі, PCA та скейлера
        model = joblib.load(f"best_model_{model_name}.pkl")
        pca = joblib.load(f"pca_model_{model_name}.pkl")
        scaler = joblib.load(f"scaler_model_{model_name}.pkl")

        # Зчитування спектра
        df = pd.read_csv(file_path, sep=";", header=0)

        # Перевірка на наявність потрібних колонок
        if "Counts" not in df.columns:
            print(f"[!] Помилка: У файлі відсутня колонка 'Counts'")
            return None

        spectrum = df["Counts"].values

        # Перевірка на довжину спектра
        if len(spectrum) != 101:  # 300-400 нм з кроком 1 нм = 101 точка
            print(f"[!] Попередження: Довжина спектра {len(spectrum)} відрізняється від очікуваної (101)")

        # Застосування скейлера
        spectrum_scaled = scaler.transform([spectrum])

        # Перетворення через PCA
        spectrum_transformed = pca.transform(spectrum_scaled)

        # Прогноз
        predicted_day = model.predict(spectrum_transformed)[0]

        print(f"Прогнозований вік молока: {predicted_day:.2f} днів")

        # Додаткова інформація про прогноз
        if predicted_day <= 0:
            print("Попередження: Негативний прогноз віку - можливо, спектр не відповідає очікуваному формату")
        elif predicted_day < 1:
            print("Молоко дуже свіже (менше 1 дня)")
        elif predicted_day <= 3:
            print("Молоко свіже (1-3 дні)")
        elif predicted_day <= 7:
            print("Молоко нормальної свіжості (4-7 днів)")
        elif predicted_day <= 10:
            print("Молоко на межі придатності (8-10 днів)")
        else:
            print("Молоко, ймовірно, непридатне до вживання (більше 10 днів)")

        return predicted_day

    except Exception as e:
        print(f"[!] Помилка при прогнозуванні: {e}")
        return None


if __name__ == "__main__":
    # Приклад використання
    file_path = input("Введіть шлях до файлу спектра (наприклад, day2_35.csv): ")
    predict_milk_age(file_path)