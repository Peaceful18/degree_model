import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # або 'Qt5Agg', якщо встановлено
import matplotlib.pyplot as plt


# Припустимо, у тебе вже є файл .csv з колонками: Wavelength, Counts
df = pd.read_csv('C:\\Users\\User\\Desktop\\degree\\degree_model\\processed_top10\\milk_ultra_pasteriz\\молоко день 17\\SpectrumFile_002.csv', sep=';')


# Побудова графіку
plt.figure(figsize=(10, 5))
plt.plot(df['Wavelength'], df['Counts'], label='Спектр', color='blue')
plt.xlabel('Довжина хвилі (нм)')
plt.ylabel('Інтенсивність (нормалізована)')
plt.title('Спектр з файлу')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

