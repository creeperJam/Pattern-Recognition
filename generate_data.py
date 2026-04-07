import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_series(n: int, mean: float = 100.0, rev_speed: float = 0.02, noise_std: float = 2.0) -> np.ndarray:
    """
    Function that generates a column of the series

    Args:
        n:         number of elemnts to generate (REQUIRED)
        mean:      base average value to use for the columns (OPTIONAL - default: 100)
        rev_speed: reversion speed (OPTIONAL - default: 0.02)
        noise_std: standard deviation for the noise generation (OPTIONAL - default: 2.0)

    Returns:
        np.ndarray: An array containing all the values of the column
    """
    column = np.empty(n)
    column[0] = mean + np.random.normal(0, noise_std)

    noise = np.random.normal(0, noise_std, n)

    for i in range(1, n):
        column[i] = column[i - 1] + rev_speed * (mean - column[i - 1]) + noise[i]

    return column


def generate_csv(output_filepath: str, num_columns: int, n_rows: int = 5_000, start_time: datetime = datetime(2024, 1, 1), freq_seconds: int = 60):
    """
    Function that generates the .csv file at the desired location

    Args:
        output_filepath: the path of the .csv file, where it should be saved (REQUIRED)
        num_columns:     the desired number of columns of the time series (OPTIONAL - default: 100)
        n_rows:          the desired number of rows of the time series (OPTIONAL - default: 100)
        start_time:      the starting date and time of the series (OPTIONAL - default: 2024-1-1)
        freq_seconds:    the amount of time that should pass between each row in seconds (OPTIONAL - default: 60)
    """
    print(f"Rows: {n_rows:,} | Columns: {num_columns}")

    ts_index = [start_time + timedelta(seconds=i * freq_seconds) for i in range(n_rows)]
    data_dict: dict = {"datetime": ts_index}

    for i in range(1, num_columns + 1):
        col = f"C{i}"

        data_dict[col] = generate_series(
            n_rows,
            mean = np.random.uniform(50, 500),
            rev_speed = np.random.uniform(0.01, 0.05),
            noise_std = np.random.uniform(1.0, 5.0),
        )

    print(f"{col} generated")

    df = pd.DataFrame(data_dict)
    df.to_csv(output_filepath, index=False, sep=";")

    print(f"\nSaved: {output_filepath}")
    print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nPreview:\n{df.head()}")
    print(f"\nStats:\n{df.drop(columns='datetime').describe().round(3)}")

if __name__ == "__main__":
    generate_csv(
        output_filepath="data.csv",
        num_columns=5,
        n_rows=2_000_000,
        freq_seconds=60,
    )