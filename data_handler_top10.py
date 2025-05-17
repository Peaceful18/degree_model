import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def calibrate_wavelengths(wavelengths):
    """Лінійне калібрування довжин хвиль."""
    return 0.73 * wavelengths + 227.7


def read_spectrum(file_path):
    """Зчитує спектр з файлу."""
    try:
        # Зчитування даних
        df = pd.read_csv(file_path, sep='\t', header=0)

        # Перейменування колонок
        if 'Nanometers' in df.columns and 'Counts' in df.columns:
            df = df.rename(columns={'Nanometers': 'Wavelength'})

        # Переконуємось, що колонки існують
        if 'Wavelength' not in df.columns or 'Counts' not in df.columns:
            print(f"Помилка: необхідні колонки не знайдені в {file_path}")
            return None

        # Калібрування
        df['Wavelength'] = calibrate_wavelengths(df['Wavelength'])

        return df

    except Exception as e:
        print(f"Помилка при зчитуванні {file_path}: {e}")
        return None


def average_background(background_files):
    """Обчислюємо середній фоновий спектр."""
    if not background_files:
        return None

    background_spectra = []
    for file in background_files:
        bg = read_spectrum(file)
        if bg is not None:
            background_spectra.append(bg)

    if not background_spectra:
        return None

    # Створюємо загальну сітку довжин хвиль
    min_wl = max([bg['Wavelength'].min() for bg in background_spectra])
    max_wl = min([bg['Wavelength'].max() for bg in background_spectra])

    # Перевіряємо, чи є діапазон
    if min_wl >= max_wl:
        print("Помилка: неправильний діапазон для фонових спектрів")
        return None

    # Створюємо рівномірну сітку для інтерполяції
    grid = np.linspace(min_wl, max_wl, 100)  # 100 точок між мін і макс

    # Інтерполюємо всі спектри на нову сітку
    interpolated = []
    for bg in background_spectra:
        interp_counts = np.interp(grid, bg['Wavelength'], bg['Counts'])
        interpolated.append(interp_counts)

    # Обчислюємо середнє
    avg_counts = np.mean(interpolated, axis=0)

    return pd.DataFrame({'Wavelength': grid, 'Counts': avg_counts})


def process_spectrum(spectrum_df, background_df, normalize=True):
    """Обробляє один спектр: віднімає фон, нормалізує і інтерполює."""
    if spectrum_df is None or background_df is None:
        return None

    # Інтерполюємо фон на сітку спектру
    bg_interp = np.interp(spectrum_df['Wavelength'],
                          background_df['Wavelength'],
                          background_df['Counts'])

    # Віднімаємо фон
    result_df = spectrum_df.copy()
    result_df['Counts'] = spectrum_df['Counts'] - bg_interp
    result_df['Counts'] = result_df['Counts'].clip(lower=0)

    # Нормалізуємо (опційно)
    if normalize:
        max_val = result_df['Counts'].max()
        if max_val > 0:
            result_df['Counts'] = result_df['Counts'] / max_val

    # Інтерполяція на фіксовану сітку
    INTERPOLATION_GRID = np.arange(442, 569, 1)
    interp_counts = np.interp(INTERPOLATION_GRID, result_df['Wavelength'], result_df['Counts'])

    return pd.DataFrame({'Wavelength': INTERPOLATION_GRID, 'Counts': interp_counts})


def process_all_spectra(base_path='data', output_path='processed_simple', normalize=True):
    """Обробка всіх спектрів з використанням простого підходу."""
    # Зчитуємо фонові спектри
    background_files = glob.glob(os.path.join(base_path, 'background', 'день 1', '*.txt'))
    if not background_files:
        print("Фонові спектри не знайдені")
        return

    print(f"Знайдено {len(background_files)} фонових файлів")

    # Створюємо середній фон
    background_df = average_background(background_files)
    if background_df is None:
        print("Не вдалося створити середній фоновий спектр")
        return

    print(f"Середній фон створено: {len(background_df)} точок")
    print(f"Діапазон фону: {background_df['Wavelength'].min():.2f} - {background_df['Wavelength'].max():.2f} нм")
    print(f"Значення фону: {background_df['Counts'].min():.2f} - {background_df['Counts'].max():.2f}")

    # Створюємо графік фону
    plt.figure(figsize=(10, 6))
    plt.plot(background_df['Wavelength'], background_df['Counts'], 'r-')
    plt.title('Середній фоновий спектр')
    plt.xlabel('Довжина хвилі (нм)')
    plt.ylabel('Інтенсивність')
    plt.grid(True)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'background.png'))
    plt.close()

    # Обробка спектрів зразків
    for milk_type in os.listdir(base_path):
        milk_path = os.path.join(base_path, milk_type)
        if not os.path.isdir(milk_path) or milk_type == 'background':
            continue

        for day_dir in glob.glob(os.path.join(milk_path, '*день*')):
            day_name = os.path.basename(day_dir)

            # Створюємо вихідну папку
            output_day_path = os.path.join(output_path, milk_type, day_name)
            os.makedirs(output_day_path, exist_ok=True)

            # Обробляємо кожен файл
            for txt_file in glob.glob(os.path.join(day_dir, '*.txt')):
                # Зчитуємо спектр
                spectrum_df = read_spectrum(txt_file)
                if spectrum_df is None:
                    print(f"Помилка при зчитуванні {txt_file}")
                    continue

                # Обробляємо спектр
                result_df = process_spectrum(spectrum_df, background_df, normalize=normalize)
                if result_df is None:
                    print(f"Помилка при обробці {txt_file}")
                    continue

                # Зберігаємо результат
                output_file = os.path.join(output_day_path, Path(txt_file).stem + '.csv')
                result_df.to_csv(output_file, index=False, sep=';')

                # Створюємо графік (для перших 5 файлів у кожній папці)
                if len(glob.glob(os.path.join(output_day_path, '*.png'))) < 5:
                    plt.figure(figsize=(12, 8))

                    plt.subplot(2, 1, 1)
                    plt.plot(spectrum_df['Wavelength'], spectrum_df['Counts'], 'b-', label='Оригінальний спектр')
                    plt.plot(background_df['Wavelength'], background_df['Counts'], 'r-', label='Фон')
                    plt.title(f'Спектр {Path(txt_file).stem}')
                    plt.xlabel('Довжина хвилі (нм)')
                    plt.ylabel('Інтенсивність')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(2, 1, 2)
                    plt.plot(result_df['Wavelength'], result_df['Counts'], 'g-')
                    plt.title('Після обробки (віднімання фону' + (' і нормалізації' if normalize else '') + ')')
                    plt.xlabel('Довжина хвилі (нм)')
                    plt.ylabel('Інтенсивність')
                    plt.grid(True)

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_day_path, Path(txt_file).stem + '.png'))
                    plt.close()

    print("Обробка завершена")


if __name__ == "__main__":
    # Запуск з нормалізацією (за замовчуванням)
    process_all_spectra(normalize=True)
