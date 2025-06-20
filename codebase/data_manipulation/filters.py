import pandas as pd
from codebase.data_manipulation.column_names import punt_buurt_code_column, punt_travel_time_column, transport_mode_col, distance_col
from codebase.data_manipulation.codebook_dicts import transport_modes_dict, trip_motives, level_mapping_suffix

def filter_by_mode_and_category(df: pd.DataFrame, mode, category):
    """
    Filter the dataframe by transport mode and trip category.
    """
    # Check if the mode and category are valid
    if mode not in transport_modes_dict.keys():
        raise ValueError(f"Invalid transport mode: {mode}. Valid modes are: {list(transport_modes_dict.keys())}")
    if category not in trip_motives.keys():
        raise ValueError(f"Invalid trip category: {category}. Valid categories are: {list(trip_motives.keys())}")

    filtered_df = df[(df["KHvm"] == mode) & (df["KMotiefV"] == category)]
    return filtered_df

def filter_by_distance_and_duration(df: pd.DataFrame, min_distance, max_distance, min_duration, max_duration):
    """
    Filter the dataframe by distance and duration.

    Distance is in 100 meters and duration is in minutes.
    """
    try:
        filtered_df = df[(df["AfstV"] >= min_distance) & (df["AfstV"] <= max_distance) &
                     (df["Reisduur"] >= min_duration) & (df["Reisduur"] <= max_duration)]
    except KeyError as e:
        raise KeyError(f"Column not found in dataframe: {e}. Ensure the dataframe contains 'AfstV' and 'Reisduur' columns.")
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


# Helper functions:
def filter_by_time(df: pd.DataFrame, max_time) -> pd.DataFrame:
    """Filters by travel time and removes duplicates based on bu_code, keeping the smallest value."""
    df_filtered: pd.DataFrame = df[df[punt_travel_time_column] <= max_time]
    df_filtered = df_filtered.sort_values(by=punt_travel_time_column, ascending=True, )
    df_filtered[punt_buurt_code_column] = df_filtered[punt_buurt_code_column].astype(str)
    df_filtered = df_filtered.drop_duplicates(subset=['bu_code'], keep='first', )
    return df_filtered