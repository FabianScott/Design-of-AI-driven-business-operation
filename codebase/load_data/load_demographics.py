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
    filename_csv = filename.replace("xlsx", "csv")
    filename_excel = filename.replace("csv", "xlsx")

    if os.path.exists(filename_csv) and filename.endswith("csv"):
        # Check if the file is a CSV file and load it directly
        df = pd.read_csv(filename_csv, sep=";", low_memory=False)
        print("Loaded file from CSV")
    else:
        df = pd.read_excel(filename_excel, engine="openpyxl", )
        df.to_csv(filename_csv, sep=";", index=False)
        print("Loaded file from Excel and saved to CSV")

    return df