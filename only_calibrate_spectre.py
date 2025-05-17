import pandas as pd
from pathlib import Path


# def calibrate_wavelengths(wavelengths):
#     # Наприклад, якщо коефіцієнти: a, b, c
#     return 8.23412698e-03 * wavelengths**2 + -5.79632937e+00 * wavelengths + 1.51065972e+03

def calibrate_wavelengths(wavelengths):
    """Лінійне калібрування довжин хвиль."""
    return 0.73 * wavelengths + 227.7


def calibrate_file(input_file, output_file=None):
    """Зчитує файл, калібрує довжини хвиль, зберігає результат."""
    try:
        # Зчитування файлу
        df = pd.read_csv(input_file, sep='\t', header=0)

        # Калібрування довжин хвиль
        df['Nanometers'] = calibrate_wavelengths(df['Nanometers'])

        # Формування вихідного імені файлу, якщо не задано явно
        if output_file is None:
            output_file = Path(input_file).with_name(Path(input_file).stem + '_calibrated.txt')

        # Збереження у форматі .txt (роздільник — табуляція)
        df.to_csv(output_file, index=False, sep='\t')

        print(f"Готово: калібрований файл збережено у {output_file}")

    except Exception as e:
        print(f"Помилка при обробці {input_file}: {e}")


if __name__ == "__main__":
    # Вкажи тут свій вхідний файл
    input_file = 'light_spectre.txt'

    # Можеш вказати output_file вручну, або залишити None — тоді створиться файл _calibrated.txt
    calibrate_file(input_file, output_file=None)
