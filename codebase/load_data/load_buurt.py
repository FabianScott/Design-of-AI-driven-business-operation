import pandas as pd


import os
import pandas as pd

def load_buurt_data(punt1, mode) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))  # locatie van dit .py bestand
    data_path = os.path.join(base_dir, "..", "..", "data", "02_punt_tot_punt_analyse", f"{punt1}_naar_buurt_{mode}.csv")
    df_punt = pd.read_csv(data_path, sep=";")
    return df_punt



#Example usage
if __name__ == "__main__":
    punt1 = "hbo_wo"
    mode = "ebike"
    df = load_buurt_data(punt1, mode)
    print(df.head())