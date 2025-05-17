import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Використовує графічний інтерфейс Tk для відображення графіків
import matplotlib.pyplot as plt
import os

# === Налаштування шляху до вхідного файлу ===
file_path = 'C:\\Users\\User\\Desktop\\degree\\degree_model\\light_spectre\\light_spectre.txt'

# === Завантаження даних ===
# Файл має бути у форматі: Nanometers<TAB>Counts
try:
    df = pd.read_csv(file_path, sep='\t')  # '\t' — роздільник табуляцією
except Exception as e:
    print(f"Не вдалося зчитати файл: {e}")
    exit()

# === Перевірка коректності колонок ===
expected_columns = ['Nanometers', 'Counts']
if not all(col in df.columns for col in expected_columns):
    print(f"У файлі мають бути колонки: {expected_columns}")
    exit()

# === Побудова спектру ===
name_spectre = 'Спектр збудження'
label_name = 'Спектр'
plt.figure(figsize=(12, 6))
plt.plot(df['Nanometers'], df['Counts'], label=label_name, color='darkblue', linewidth=1)

# === Оформлення графіка ===
plt.title(name_spectre, fontsize=14)
plt.xlabel('Довжина хвилі (нм)', fontsize=12)
plt.ylabel('Інтенсивність сигналу', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# === Відображення графіка ===
plt.show()
