import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib

# === Шлях до даних ===
root_dir = "processed_top10"
milk_dirs = ["milk_one_pasteriz", "milk_two_pasteriz"]

X = []
y = []

# === Збір спектрів ===
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

# === Перевірка на дані ===
if len(X) == 0:
    raise ValueError("Жодного спектра не зчитано!")

# === Розбиття на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Побудова пайплайну ===
pipeline = Pipeline([
    ('pca', PCA()),  # Спочатку PCA
    ('regressor', RandomForestRegressor(random_state=42))  # Потім модель
])

# === Параметри для підбору ===
param_grid = {
    'pca__n_components': [10, 20, 30],  # Скільки головних компонент залишити
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, None],
}

# === GridSearchCV ===
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)

# === Навчання з пошуком параметрів ===
grid_search.fit(X_train, y_train)

# === Найкраща модель ===
best_model = grid_search.best_estimator_

# === Оцінка ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Результати найкращої моделі ===")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print("Найкращі параметри:", grid_search.best_params_)

# # === Збереження моделі ===
# joblib.dump(best_model, 'best_rf_pca_model.pkl')

# === Візуалізація ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Справжні значення (дні)")
plt.ylabel("Прогнозовані значення")
plt.title("Прогноз з PCA + RandomForest")
plt.grid(True)
plt.show()
