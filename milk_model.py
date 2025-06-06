import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import joblib

# === Завантаження спектрів ===
root_dir = "processed_top10"
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

# === Масштабування даних ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === PCA ===
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# === Вибір моделі ===
model_name = 'random_forest'

if model_name == 'random_forest':
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt']
    }
elif model_name == 'xgboost':
    model = XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
    }
elif model_name == 'lightgbm':
    model = LGBMRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [15, 31]
    }
else:
    raise ValueError("Невідома модель")

# === Підбір параметрів ===
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_pca, y_train)

best_model = grid_search.best_estimator_

# === Прогноз і оцінка ===
y_pred = best_model.predict(X_test_pca)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== {model_name.upper()} РЕЗУЛЬТАТИ ===")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print("Найкращі параметри:", grid_search.best_params_)

# === Збереження ===
joblib.dump(best_model, f'best_model_{model_name}.pkl')
joblib.dump(pca, f'pca_model_{model_name}.pkl')
joblib.dump(scaler, f'scaler_model_{model_name}.pkl')
print(f"Модель збережено як best_model_{model_name}.pkl")

# === Візуалізація ===
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Справжній день")
plt.ylabel("Прогнозований день")
plt.title(f"Прогноз ({model_name.upper()} + PCA + StandardScaler)")
plt.grid(True)
plt.tight_layout()
plt.show()
