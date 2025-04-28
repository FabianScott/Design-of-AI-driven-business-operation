import os
import pandas as pd


def load_demograhics(demographics_filename = "data/demographics/kwb-2023.csv"):
    """Wrapper for loading the demographics data."""
    return load_excel(demographics_filename)

def load_excel(filename):
    """
    Load an excel file and save it as a CSV file for faster loading in the future.
    If the CSV file already exists, load it instead of the excel file.
    """

    if os.path.exists(filename) and filename.endswith(".csv"):
        # Check if the file is a CSV file and load it directly
        df = pd.read_csv(filename, sep=";", low_memory=False)
        print("Loaded file from CSV")
    else:
        df = pd.read_excel(filename.replace("csv", "xlsx"), )
        df.to_csv(filename, sep=";", index=False)
        print("Loaded file from Excel and saved to CSV")

    return df