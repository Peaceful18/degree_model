import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib

# === Шлях до твоїх даних ===
root_dir = "processed_top10"
milk_dirs = ["milk_one_pasteriz", "milk_two_pasteriz"]

X = []
y = []

# === Збір даних ===
for milk_dir in milk_dirs:
    full_path = os.path.join(root_dir, milk_dir)
    for day_folder in os.listdir(full_path):
        if not day_folder.startswith("день"):
            continue
        try:
            day_number = int(day_folder.split(" ")[1])
        except:
            print(f"Пропущено: {day_folder}")
            continue

        day_path = os.path.join(full_path, day_folder)
        for filename in os.listdir(day_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(day_path, filename)
                try:
                    df = pd.read_csv(file_path, sep=";", header=0)
                    if df.shape[1] != 2:
                        continue
                    spectrum = df["Counts"].values
                    X.append(spectrum)
                    y.append(day_number)
                except Exception as e:
                    print(f"Помилка з файлом {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Кількість зчитаних спектрів: {len(X)}")

# === Перевірка на наявність даних ===
if len(X) == 0:
    raise ValueError("Жодного спектра не зчитано. Перевір шлях або формат CSV!")

# === Розбиття ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Параметри для RandomizedSearch ===
param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

# === Базова модель ===
base_model = RandomForestRegressor(random_state=42)

# === RandomizedSearchCV ===
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=30,  # Скільки випадкових комбінацій перевірити
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# === Навчання з RandomizedSearchCV ===
random_search.fit(X_train, y_train)

# === Найкраща модель ===
best_model = random_search.best_estimator_

# === Оцінка ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Результати найкращої моделі ===")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print("Найкращі параметри:", random_search.best_params_)

# === Збереження моделі ===
joblib.dump(best_model, 'best_rf_model_random.pkl')
print("Модель збережено у 'best_rf_model_random.pkl'")

# === Побудова графіка
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Справжні значення (дні)")
plt.ylabel("Прогнозовані значення")
plt.title("Прогноз RandomForest після RandomizedSearchCV")
plt.grid(True)
plt.show()
