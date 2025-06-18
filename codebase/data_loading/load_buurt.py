import pandas as pd
from tqdm import tqdm


def load_buurt_data(punt1, mode) -> pd.DataFrame:
    """Load the punt data for the specified mode."""
    df_punt = pd.read_csv(f"data/02_punt_tot_punt_analyse/{punt1}_naar_buurt_{mode}.csv", sep=";")
    df_punt = df_punt.dropna()
    return df_punt


def read_all_punt_to_punt(punten, modes):
    """
    Reads all punt to punt files defined in the list of punt names and modes.
    """
    matrix_data = {}

    for punt in tqdm(punten, desc="Loading punt to punt data"):
        for mode in modes:
            key = f"{punt}_{mode}"
            try:
                df = load_buurt_data(punt, mode)
                matrix_data[key] = df.copy()
            except Exception as e:
                print(f"Skipping {key}: {e}")

    return matrix_data