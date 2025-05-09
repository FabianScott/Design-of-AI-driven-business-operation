import pandas as pd
from codebase.data.column_names import transport_mode_col, distance_col
from codebase.data.codebook_dicts import transport_modes, trip_motives, level_mapping_suffix

def filter_by_mode_and_category(df: pd.DataFrame, mode, category):
    """
    Filter the dataframe by transport mode and trip category.
    """
    # Check if the mode and category are valid
    if mode not in transport_modes.keys():
        raise ValueError(f"Invalid transport mode: {mode}. Valid modes are: {list(transport_modes.keys())}")
    if category not in trip_motives.keys():
        raise ValueError(f"Invalid trip category: {category}. Valid categories are: {list(trip_motives.keys())}")

    filtered_df = df[(df["KHvm"] == mode) & (df["KMotiefV"] == category)]
    return filtered_df

def filter_by_distance_and_duration(df: pd.DataFrame, min_distance, max_distance, min_duration, max_duration):
    """
    Filter the dataframe by distance and duration.

    Distance is in 100 meters and duration is in minutes.
    """
    filtered_df = df[(df["AfstV"] >= min_distance) & (df["AfstV"] <= max_distance) &
                     (df["Reisduur"] >= min_duration) & (df["Reisduur"] <= max_duration)]
    return filtered_df

def filter_by_origin(df: pd.DataFrame, origins, level=0):
    """
    Filter the dataframe by origin.
    Level is 0 (Buurt), 1 (Geemente), 2 (Province)
    """
    if level not in level_mapping_suffix.keys():
        raise ValueError(f"Invalid level: {level}. Valid levels are: {list(level_mapping_suffix.keys())}")
    
    filtered_df = df[df["Vert" + level_mapping_suffix[level]].isin(origins)]
    return filtered_df
    
def filter_by_destination(df: pd.DataFrame, destinations, level=0):
    """
    Filter the dataframe by destination.
    Level is 0 (Buurt), 1 (Geemente), 2 (Province)
    """
    if level not in level_mapping_suffix.keys():
        raise ValueError(f"Invalid level: {level}. Valid levels are: {list(level_mapping_suffix.keys())}")
    
    filtered_df = df[df["Aank" + level_mapping_suffix[level]].isin(destinations)]
    return filtered_df

def filter_by_motive(df: pd.DataFrame, motives):
    """
    Filter the dataframe by motive.
    """
    filtered_df = df[df["KMotiefV"].isin(motives)]
    return filtered_df