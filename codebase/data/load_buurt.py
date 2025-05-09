import pandas as pd


def load_buurt_data(punt1, mode) -> pd.DataFrame:
    """Load the punt data for the specified mode."""
    df_punt = pd.read_csv(f"data/02_punt_tot_punt_analyse/{punt1}_naar_buurt_{mode}.csv", sep=";")
    df_punt = df_punt.dropna()
    return df_punt