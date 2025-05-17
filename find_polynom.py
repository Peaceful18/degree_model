import numpy as np
import matplotlib.pyplot as plt

# Виміряні спектрометром (старі)
measured = np.array([359, 434, 455])

# Справжні значення (ртутна лампа)
true = np.array([491, 546, 578])

# Побудова полінома 2-го ступеня
coeffs = np.polyfit(measured, true, 2)
print("Коефіцієнти калібрування:", coeffs)

# Створення функції калібрування
calibration_func = np.poly1d(coeffs)

# Візуалізація (опціонально)
x_vals = np.linspace(min(measured)-10, max(measured)+10, 200)
y_vals = calibration_func(x_vals)

plt.plot(x_vals, y_vals, label="Калібрувальна функція")
plt.scatter(measured, true, color='red', label="Контрольні точки")
plt.xlabel("Виміряна довжина хвилі")
plt.ylabel("Скоригована довжина хвилі")
plt.title("Калібрування довжин хвиль")
plt.legend()
plt.grid(True)
plt.show()
