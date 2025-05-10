import os
import pandas as pd
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parents[2]


from pathlib import Path
from codebase.utils.path_utils import get_project_root
import pandas as pd

def load_excel(filename_rel_to_root):
    project_root = get_project_root()
    filename_csv = project_root / filename_rel_to_root
    filename_excel = filename_csv.with_suffix(".xlsx")

    if filename_csv.exists():
        df = pd.read_csv(filename_csv, sep=";", low_memory=False)
        print(f"Loaded file from CSV: {filename_csv}")
    elif filename_excel.exists():
        df = pd.read_excel(filename_excel, engine="openpyxl")
        df.to_csv(filename_csv, sep=";", index=False)
        print(f"Loaded from Excel and saved to CSV: {filename_excel}")
    else:
        raise FileNotFoundError(f"Neither {filename_csv} nor {filename_excel} could be found.")

    return df

def load_demograhics():
    return load_excel("data/demographics/kwb-2023.csv")


df = load_demograhics()
print(df.head())

