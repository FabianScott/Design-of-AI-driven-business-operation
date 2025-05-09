import pandas as pd
from codebase.load_data.column_names import transport_mode_col, distance_col

# KHvm
transport_modes = {
    1: "Car - driver",
    2: "Car - passenger",
    3: "Train",
    4: "Bus/tram/metro",
    5: "Bicycle",
    6: "On foot",
    7: "Other"
}
# KMotiefV
trip_motives = {
    1: "Work",      # Commute to/from work
    2: "Business",  # Business visit (work-related)
    3: "Services/personal care",
    4: "Shopping",  # Groceries
    5: "Education", # Course
    6: "Visit/stay overnight",
    7: "Other social/recreational",
    8: "Touring/walking",
    9: "Other motive"
}

level_mapping_suffix = {
    0: "PC",
    1: "Gem",
    2: "Prov"
}

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