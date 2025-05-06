import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

INTERPOLATION_GRID = np.arange(300, 401, 1)  # сітка 300–400 нм з кроком 1 нм

def read_and_clean_spectrum(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    has_header = 'Count' in first_line or 'Wave' in first_line or 'Nano' in first_line

    df = pd.read_csv(file_path, sep='\t', header=0 if has_header else None)

    df.columns = [col.strip() for col in df.columns]
    rename_map = {}
    for col in df.columns:
        if 'nano' in col.lower() or 'wave' in col.lower():
            rename_map[col] = 'Wavelength'
        elif 'count' in col.lower():
            rename_map[col] = 'Counts'
    df = df.rename(columns=rename_map)

    if 'Wavelength' not in df.columns or 'Counts' not in df.columns:
        raise ValueError(f"[!] Немає колонок 'Wavelength' і 'Counts' у файлі: {file_path}")

    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    df['Counts'] = pd.to_numeric(df['Counts'], errors='coerce')
    df = df.dropna()
    df = df[df['Counts'] > 0].copy()
    return df


def normalize_spectrum(df):
    max_val = df['Counts'].max()
    if max_val == 0:
        return df.copy()
    df = df.copy()
    df['Counts'] = df['Counts'] / max_val
    return df


def interpolate_to_grid(df, grid):
    if df['Wavelength'].min() > grid[0] or df['Wavelength'].max() < grid[-1]:
        return None  # спектр не перекриває повністю діапазон — пропускаємо
    interpolated_counts = np.interp(grid, df['Wavelength'], df['Counts'])
    return pd.DataFrame({'Wavelength': grid, 'Counts': interpolated_counts})


def average_background(background_files):
    spectra = []
    for file in background_files:
        df = read_and_clean_spectrum(file)
        df = normalize_spectrum(df)
        df_interp = interpolate_to_grid(df, INTERPOLATION_GRID)
        if df_interp is not None:
            spectra.append(df_interp['Counts'].values)

    if not spectra:
        return None

    avg_counts = np.mean(spectra, axis=0)
    return pd.DataFrame({'Wavelength': INTERPOLATION_GRID, 'Counts': avg_counts})


def process_day(input_day_path, background_df, output_day_path):
    os.makedirs(output_day_path, exist_ok=True)

    for txt_file in glob.glob(os.path.join(input_day_path, '*.txt')):
        try:
            df = read_and_clean_spectrum(txt_file)
            df = normalize_spectrum(df)
            df_interp = interpolate_to_grid(df, INTERPOLATION_GRID)
            if df_interp is None:
                print(f"[!] Пропущено {txt_file} — не перекриває 300–400 нм")
                continue

            df_interp['Counts'] = df_interp['Counts'] - background_df['Counts']
            df_interp.to_csv(os.path.join(output_day_path, Path(txt_file).stem + '.csv'), index=False, sep=';')
        except Exception as e:
            print(f"[!] Помилка з файлом {txt_file}: {e}")


def process_all(base_path='data', output_path='processed_max'):
    background_files = glob.glob(os.path.join(base_path, 'background', 'день 1', '*.txt'))
    if not background_files:
        print("[!] Немає фонів у background/day 1 — обробка припинена")
        return

    background_df = average_background(background_files)
    if background_df is None:
        print("[!] Порожній фон у background/day 1 — обробка припинена")
        return

    for milk_type in os.listdir(base_path):
        milk_path = os.path.join(base_path, milk_type)
        if not os.path.isdir(milk_path) or milk_type == 'background':
            continue

        for day_dir in glob.glob(os.path.join(milk_path, '*день*')):
            day_name = os.path.basename(day_dir)
            output_day_path = os.path.join(output_path, milk_type, day_name)
            process_day(day_dir, background_df, output_day_path)


if __name__ == '__main__':
    process_all()
    print("[*] Обробка завершена")
