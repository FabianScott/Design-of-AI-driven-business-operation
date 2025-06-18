import os
import numpy as np
import pandas as pd
import sys



from codebase.data_loading.load_demographics import load_excel
from codebase.data_manipulation.column_lists import (
    numerical_cols, ordinal_cols, categorical_cols, binary_cols
)

import os
import pandas as pd

def load_odin(years=None, only_one_mode=True, do_apply_ignore_rules=True, dropna=False):
    """
    Reads and concatenates ODiN data for the specified years via load_excel().
    Defaults to [2019, 2020, 2021, 2022, 2023] if no `years` provided.
    Returns:
        odin_df (DataFrame): concatenated ODiN data.
    """
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023]

    print(f"[INFO] Loading ODiN data for years: {years}")

    odin_dfs = []
    for year in years:
        base_folder = os.path.join("data", "OdiN 2019-2023", f"OdiN {year}")
        filename = f"ODiN{year}_Databestand.csv"
        if year in [2019, 2020]:
            filename = filename.replace("Databestand", "Databestand_v2.0")
        odin_path = os.path.join(base_folder, filename)

        print(f"[INFO] Reading file: {odin_path}")
        df_year = load_excel(odin_path)
        print(f"[INFO] Loaded {len(df_year):,} rows for {year}")

        odin_dfs.append(df_year)

    odin_df = pd.concat(odin_dfs, ignore_index=True)
    print(f"[INFO] Total rows after concatenation: {len(odin_df):,}")

    if only_one_mode:
        before = len(odin_df)
        odin_df = odin_df[odin_df["Verpl"] == 1]
        after = len(odin_df)
        print(f"[FILTER] Only-one-mode trips: {after:,} rows (filtered {before - after:,})")

    if do_apply_ignore_rules:
        before = len(odin_df)
        odin_df = apply_ignore_rules(odin_df, IGNORE_RULES)
        after = len(odin_df)
        print(f"[FILTER] After ignore rules: {after:,} rows (filtered {before - after:,})")

    if dropna:
        before_cols = odin_df.shape[1]
        odin_df = odin_df.dropna(axis=1, how='any')
        after_cols = odin_df.shape[1]
        print(f"[CLEAN] Dropped columns with NaNs: {before_cols - after_cols} columns removed")

    print(f"[DONE] Final dataset shape: {odin_df.shape}")
    return odin_df


def odin_add_buurtcode(odin_df, mapping_path="data/buurt_to_PC_mapping.csv", buurt_code_column="BuurtCode"):
    """
    Adds a 'BuurtCode' column to odin_df by matching the first 4 digits of 'WoPC'
    to the first 4 digits of 'PC6' in the mapping CSV.
    """
    mapping_df = pd.read_csv(mapping_path, dtype={"PC6": str}, low_memory=False)
    mapping_df["PC4"] = mapping_df["PC6"].str[:4]

    result_df = (
        mapping_df[["Buurt2024", "PC4"]]
        .drop_duplicates()
        .rename(columns={"PC4": "PC6"})
        .drop_duplicates(subset=["PC6"], keep="first")
        .reset_index(drop=True)
    )

    odin_df = odin_df.copy()
    odin_df["WoPC"] = odin_df["WoPC"].astype(str)
    odin_df["WoPC4"] = odin_df["WoPC"].str[:4]

    mapping_dict = result_df.set_index("PC6")["Buurt2024"].to_dict()
    odin_df[buurt_code_column] = odin_df["WoPC4"].map(mapping_dict)
    odin_df.drop(columns=["WoPC4"], inplace=True)

    return odin_df


def clean_aggregate_numord(odin_df, numerical_cols, ordinal_cols):
    """
    Cleans & aggregates numerical + ordinal columns by BuurtCode.
    Returns a DataFrame of mean values per BuurtCode.
    """
    exact_ignore_map = {
        **{col: [11] for col in ["HHPers", "HHLft1", "HHLft2", "HHLft3", "HHLft4"]},
        **{col: [9994, 9995] for col in ["BouwjaarPa1", "BouwjaarPa2", "BouwjaarPaL"]},
        "BetWerk": [4, 5],
        **{col: [8, 9] for col in ["KBouwjaarPa1", "KBouwjaarPa2", "KBouwjaarPaL"]},
        "SAntAdr": [7],
        "HHLaagInk": [9],
        "HHSocInk": [10],
        **{col: [11] for col in ["HHBestInkG", "HHGestInkG", "HHWelvG"]}
    }
    range_threshold_map = {
        **{col: 10 for col in [
            "HHRijbewijsAu", "HHRijbewijsMo", "HHRijbewijsBr",
            "HHAuto", "HHAutoL", "OPAuto", "HHMotor", "OPMotor",
            "HHBrom", "OPBrom", "HHSnor", "OPSnor"
        ]},
        **{col: 1 for col in ["AfstandOP", "AfstandSOP", "AfstV", "AfstR", "AfstRBL"]},
        "KAfstV": 1, "KAfstR": 1,
        **{col: 6 for col in [
            "KGewichtPa1", "KGewichtPa2", "KGewichtPaL",
            "BerHalte", "BerFam", "BerSport", "BerWrk", "BerOnd",
            "BerSup", "BerZiek", "BerArts", "BerStat"
        ]}
    }

    cols = [c for c in (numerical_cols + ordinal_cols) if c in odin_df.columns] + ["BuurtCode"]
    df = odin_df[cols].copy()
    df["BuurtCode"] = df["BuurtCode"].astype(str)

    # Exact→NaN
    replace_dict = {
        col: {val: pd.NA for val in vals}
        for col, vals in exact_ignore_map.items() if col in df.columns
    }
    if replace_dict:
        df.replace(replace_dict, inplace=True)

    # Range masks
    for col, thresh in range_threshold_map.items():
        if col not in df.columns:
            continue
        if thresh == 1:
            df.loc[df[col] < 1, col] = pd.NA
        else:
            df.loc[df[col] >= thresh, col] = pd.NA

    # Coerce to numeric
    for col in (numerical_cols + ordinal_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["BuurtCode"].notna()]
    valid_cols = [c for c in (numerical_cols + ordinal_cols) if c in df.columns]
    return df.groupby("BuurtCode", observed=True)[valid_cols].mean().reset_index()


def clean_aggregate_categorical(odin_df, categorical_cols):
    """
    Cleans & aggregates categorical columns by BuurtCode (mode).
    Returns a DataFrame of mode values per BuurtCode.
    """
    cols = [c for c in categorical_cols if c in odin_df.columns] + ["BuurtCode"]
    df = odin_df[cols].copy()
    df["BuurtCode"] = df["BuurtCode"].astype(str)
    df = df[df["BuurtCode"].notna()]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_cols = [c for c in categorical_cols if c in df.columns]
    return (
        df.groupby("BuurtCode", observed=True)[valid_cols]
          .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
          .reset_index()
    )


def clean_aggregate_binary(odin_df, binary_cols):
    """
    Cleans & aggregates binary columns by BuurtCode (probability of 1).
    Returns a DataFrame of mean values per BuurtCode.
    """
    cols = [c for c in binary_cols if c in odin_df.columns] + ["BuurtCode"]
    df = odin_df[cols].copy()
    df["BuurtCode"] = df["BuurtCode"].astype(str)
    df = df[df["BuurtCode"].notna()]

    groupA = [
        "WrkVerg", "MeerWink", "OPRijbewijsAu", "OPRijbewijsMo", "OPRijbewijsBr",
        "HHEFiets", "Kind6", "CorrVerpl", "SDezPlts", "Toer",
        "VergVast", "VergKm", "VergBrSt", "VergOV",
        "VergAans", "VergVoer", "VergBudg", "VergPark", "VergStal", "VergAnd"
    ]
    groupB = ["ByzAdr", "ByzVvm", "ByzTyd", "ByzDuur", "ByzRoute"]

    for col in groupA:
        if col in df.columns:
            df.loc[~df[col].isin([0, 1]), col] = pd.NA

    for col in groupB:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})

    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_cols = [c for c in binary_cols if c in df.columns]
    return df.groupby("BuurtCode", observed=True)[valid_cols].mean().reset_index()


def merge_odin_stats(demographics, odin_stats, key_demog="gwb_code_8", key_odin="BuurtCode"):
    """
    Merges odin_stats into demographics on demographics[key_demog] ↔ odin_stats[key_odin].
    Returns merged DataFrame.
    """
    odin_stats[key_odin] = odin_stats[key_odin].astype(str)
    demographics[key_demog] = demographics[key_demog].astype(str)

    merged = pd.merge(
        demographics,
        odin_stats,
        left_on=key_demog,
        right_on=key_odin,
        how="left",
        validate="one_to_one"
    )
    merged.drop(columns=[key_odin], inplace=True)
    return merged


def prepare_odin_stats(odin_df, buurt_code_column="BuurtCode"):
    """
    Orchestrates cleaning & aggregation across all variable types.
    Returns a single odin_stats DataFrame indexed by BuurtCode.
    """

    means = clean_aggregate_numord(odin_df, numerical_cols, ordinal_cols)
    modes = clean_aggregate_categorical(odin_df, categorical_cols)
    probs = clean_aggregate_binary(odin_df, binary_cols)

    temp = pd.merge(means, modes, on=buurt_code_column, how="outer", validate="one_to_one")
    odin_stats = pd.merge(temp, probs, on=buurt_code_column, how="outer", validate="one_to_one")

    counts = odin_df[buurt_code_column].value_counts().rename_axis(buurt_code_column).reset_index(name='Count')
    odin_stats = pd.merge(odin_stats, counts, on=buurt_code_column, how="left", validate="one_to_one")

    return odin_stats


def apply_ignore_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Replace values that should be ignored by NaN.

    Parameters
    ----------
    df     : DataFrame to clean (modified in place and returned)
    rules  : {col: iterable   -> every element in iterable set to NaN
              col: callable   -> rows where callable(series) is True set to NaN}

    Returns
    -------
    DataFrame  (same object, for chaining)
    """
    for col, rule in rules.items():
        if col not in df.columns:
            continue                    # just skip missing columns
        if callable(rule):
            mask = rule(df[col])
        else:                           # treat it as a list / set of values
            mask = df[col].isin(rule)
        df.loc[mask, col] = np.nan
    return df


# Codes that are 'unknown', 'N/A' or otherwise unusable
IGNORE_RULES = {
    # ---- household counters: ignore exactly 11  ---------------------------
    **{c: [11] for c in ["HHPers", "HHLft1", "HHLft2", "HHLft3", "HHLft4"]},

    # ---- licence / vehicle counters: ignore >=10  -------------------------
    **{c: (lambda s: s >= 10) for c in [
            "HHRijbewijsAu","HHRijbewijsMo","HHRijbewijsBr",
            "HHFiets","HHAuto","HHAutoL","OPAuto","HHMotor",
            "OPMotor","HHBrom","OPBrom","HHSnor","OPSnor",
            "RAantIn"]},

    # ---- vehicle year: ignore 9994 and 9995  ------------------------------
    **{c: [9994, 9995] for c in ["BouwjaarPa1","BouwjaarPa2","BouwjaarPaL"]},

    # ---- 0 means ‘no displacement’: set to NaN so it is treated as missing
    **{c: [0] for c in [
            "ReisduurOP","AfstandOP","AfstandSOP","AfstV",
            "AfstR","AfstRBL","RReisduur","RReisduurBL"]},

    # ----------------------------------------------------------------------
    # BINARY columns that must be 0/1 only  → everything else ⇒ NaN
    # ----------------------------------------------------------------------
    **{c: (lambda s: ~s.isin([0, 1])) for c in [
            "WrkVerg","MeerWink","OPRijbewijsAu","OPRijbewijsMo",
            "OPRijbewijsBr","HHEFiets","Kind6","CorrVerpl","SDezPlts",
            "Toer","VergVast","VergKm","VergBrSt","VergOV",
            "VergAans","VergVoer","VergBudg","VergPark",
            "VergStal","VergAnd"]},

    # ----------------------------------------------------------------------
    # ‘1 = Yes / 2 = No’ blocks  → leave only 1 or 2
    # ----------------------------------------------------------------------
    **{c: (lambda s: ~s.isin([1, 2])) for c in [
            "AutoEig","AutoHhl","AutoLWg","AutoLPl","AutoBed",
            "AutoDOrg","AutoDPart","AutoDBek","AutoLeen","AutoHuur",
            "AutoAnd","ByzAdr","ByzVvm","ByzTyd","ByzDuur","ByzRoute"]},

    # ----------------------------------------------------------------------
    # Ordinal columns with one or a few ‘unknown’ codes
    # (examples below; extend freely)
    # ----------------------------------------------------------------------
    "BetWerk"     : [4, 5],
    "KBouwjaarPa1": [8, 9],  "KBouwjaarPa2": [8, 9],  "KBouwjaarPaL": [8, 9],
    "KGewichtPa1" : (lambda s: s >= 6),   # 6,7 are unknown/N.A.
    "SAantAdr"    : [7],
    "HHLaagInk"   : [9],
    "HHSocInk"    : [10],
    "HHBestInkG"  : [11],  "HHGestInkG": [11],  "HHWelvG": [11],
    "RTSamen"     : (lambda s: s >= 12),
}

    