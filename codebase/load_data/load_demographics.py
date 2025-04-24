import os
import pandas as pd


def load_demograhics(demographics_filename = "data/demographics/kwb-2023.csv"):

    if os.path.exists(demographics_filename):
        df_demographics = pd.read_csv(demographics_filename, sep=";", low_memory=False)
        print("Loaded demographics from CSV")
    else:
        df_demographics = pd.read_excel(demographics_filename.replace("csv", "xlsx"), )
        df_demographics.to_csv(demographics_filename, sep=";", index=False)
        print("Loaded demographics from Excel and saved to CSV")

    return df_demographics