import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib

# === Завантаження спектрів ===
root_dir = "processed_simple"
milk_dirs = ["milk_one_pasteriz", "milk_two_pasteriz"]

X = []
y = []

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
if len(X) == 0:
    raise ValueError("Жодного спектра не зчитано!")

# === Розбиття ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === PCA ===
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# === LightGBM ===
model = LGBMRegressor(random_state=42, n_estimators=1000)

param_dist = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.005, 0.01, 0.02],
    'num_leaves': [7, 10, 15],
    'reg_alpha': [0.0, 1.0, 5.0],
    'reg_lambda': [0.0, 1.0, 5.0],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# === Randomized Search ===
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# === Правильний спосіб early_stopping через callbacks ===
fit_params = {
    "eval_set": [(X_test_pca, y_test)],
    "eval_metric": "mae",
    "callbacks": [early_stopping(stopping_rounds=20)]
}

# === Навчання
random_search.fit(X_train_pca, y_train, **fit_params)

best_model = random_search.best_estimator_

# === Прогноз і оцінка ===
y_pred = best_model.predict(X_test_pca)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== LIGHTGBM РЕЗУЛЬТАТИ ===")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print("Найкращі параметри:", random_search.best_params_)

# Оцінка на train
y_train_pred = best_model.predict(X_train_pca)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Train MAE: {train_mae:.2f}, Train R²: {train_r2:.2f}")
print(f"Test  MAE: {mae:.2f}, Test  R²: {r2:.2f}")

# === Збереження ===
joblib.dump(best_model, f'best_model_lightgbm.pkl')
joblib.dump(pca, f'pca_model_lightgbm.pkl')
print(f"Модель збережено як best_model_lightgbm.pkl")

# === Візуалізація ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Справжній день")
plt.ylabel("Прогнозований день")
plt.title(f"Прогноз (LIGHTGBM + PCA)")
plt.grid(True)
plt.tight_layout()
plt.show()
