import pandas as pd
from codebase.data.load_demographics import load_demograhics
from .willingness import willingness_to_cycle
from codebase.data.load_buurt import load_buurt_data
from codebase.data.column_names import (
    demographics_population_column,
    demographics_buurt_code_column,
    punt_travel_time_column,
    punt_buurt_code_column,
    willingness_to_cycle_column,
)


# Helper functions:
def filter_by_time(df: pd.DataFrame, max_time) -> pd.DataFrame:
    """Filters by travel time and removes duplicates based on bu_code, keeping the smallest value."""
    df_filtered: pd.DataFrame = df[df[punt_travel_time_column] <= max_time]
    df_filtered = df_filtered.sort_values(by=punt_travel_time_column, ascending=True, )
    df_filtered[punt_buurt_code_column] = df_filtered[punt_buurt_code_column].astype(str)
    df_filtered = df_filtered.drop_duplicates(subset=['bu_code'], keep='first', )
    return df_filtered

def get_buurt_ids(df: pd.DataFrame) -> list:
    df_buurt = df[[punt_buurt_code_column]].astype(str)
    return df_buurt.values.flatten().tolist()

def add_willingness_to_cycle_column(df: pd.DataFrame, location: str, mode="fiets", travel_time_col=punt_travel_time_column, willingness_col=willingness_to_cycle_column) -> pd.DataFrame:
    willingness_array = willingness_to_cycle(df[travel_time_col], location, mode=mode)
    df_filtered = df.copy()
    df_filtered[willingness_col] = willingness_array
    return df_filtered

# Function to calculate the total number of inhabitants willing to cycle within a certain time frame
def get_total_inhabitants_in_buurts(df, within_mins, df_demographics=None):
    df_filtered = filter_by_time(df, within_mins)
    df_demographics = load_demograhics() if df_demographics is None else df_demographics

    buurt_ids = get_buurt_ids(df_filtered)
    df_location = df_demographics[df_demographics[demographics_buurt_code_column].astype(str).isin(buurt_ids)]
    total_inhabitants = df_location[demographics_population_column].sum()
    return total_inhabitants

def align_by_buurt(df_filtered, df_demographics) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns the filtered dataframe with the demographics dataframe by buurt code.
    """
    buurt_ids = get_buurt_ids(df_filtered)
    df_location = df_demographics[df_demographics[demographics_buurt_code_column].astype(str).isin(buurt_ids)]

    if len(df_location) != len(df_filtered):
        print(f"Demographics and filtered dataframes do not match in length: {len(df_location)} vs {len(df_filtered)}, ignoring missing values")

    s1 = set(df_filtered[punt_buurt_code_column].unique())
    s2 = set(df_location[demographics_buurt_code_column].unique())
    # Only keep the rows in both dataframes that have matching buurt codes
    df_filtered = df_filtered[df_filtered[punt_buurt_code_column].isin(s1.intersection(s2))]
    df_location = df_location[df_location[demographics_buurt_code_column].isin(s1.intersection(s2))]
    # Sort the dataframes by the buurt code to ensure they match
    df_filtered = df_filtered.sort_values(by=punt_buurt_code_column)
    df_location = df_location.sort_values(by=demographics_buurt_code_column)

    df_filtered.reset_index(drop=True, inplace=True)
    df_location.reset_index(drop=True, inplace=True)

    return df_filtered, df_location

def get_total_willingness_to_cycle_in_buurts(df: pd.DataFrame, location, within_mins, mode, df_demographics=None, ):
    df_demographics = load_demograhics() if df_demographics is None else df_demographics
    df_filtered = filter_by_time(df, within_mins)
    # Get the unique buurt codes from the filtered dataframe
    df_filtered = add_willingness_to_cycle_column(df_filtered, location, mode=mode)
    # Filter the demographics dataframe to only include the relevant buurt codes
    df_filtered, df_location = align_by_buurt(df_filtered, df_demographics)
    total_willingness = int((df_filtered[willingness_to_cycle_column].values * df_location[demographics_population_column].values).sum())
    return total_willingness

# Main function for calculating the total number of inhabitants and willingness to cycle
def get_total_inhabitants_and_willingness(punt1, mode, within_mins, location="Elementary Schools", verbose=False):
    """
    Function to load the demographics data, filter the punt data by travel time,
    and calculate the total number of inhabitants and willingness to cycle within the specified time.

    Args:
        punt1 (str): The first point of interest, not "buurt". Second one is always "buurt".
        mode (str): The mode of transport, one of "fiets" or "ebike"
        within_mins (int): The maximum travel time in minutes.
        location (str): The location for which to calculate willingness to cycle.
        verbose (bool): If True, prints additional information.
    Returns:
        total_inhabitants (int): The total number of inhabitants within the specified time.
        total_willing_cyclists (int): The total number of willing cyclists within the specified time.
    """

    df_demographics = load_demograhics()
    df_punt = load_buurt_data(punt1, mode=mode)
    nl_total = df_demographics[demographics_population_column][0]
    
    total_inhabitants = get_total_inhabitants_in_buurts(df_punt, within_mins=within_mins, df_demographics=df_demographics)
    total_willing_cyclists = get_total_willingness_to_cycle_in_buurts(
        df_punt, 
        location=location, 
        mode=mode,
        within_mins=within_mins, 
        df_demographics=df_demographics
        )
    
    if verbose:
        print(f"Total inhabitants within {within_mins} minutes of {punt1} from buurt: {total_inhabitants} of {nl_total} = {total_inhabitants/nl_total:.2%} of the Netherlands")
        print(f"Total willingness to cycle of those: {total_willing_cyclists} of {total_inhabitants} = {total_willing_cyclists/total_inhabitants:.2%}")
    
    return total_inhabitants, total_willing_cyclists


def number_of_residents_in_detour(detour_factor, punt, mode, within_mins):
    """
    Returns a DataFrame of neighborhoods (bu_codes) where the detour factor exceeds a given threshold,
    along with the population.
    """
    demographics = load_demograhics()
    df_punt = load_buurt_data(punt, mode)
    df = filter_by_time(df_punt, within_mins)

    high_detour = df[df['omrijdfactor'] > detour_factor]

    merged_df = high_detour.merge(
        demographics[['gwb_code', 'a_inw']], 
        how='left',
        left_on='bu_code',
        right_on='gwb_code'
    )

    merged_df.drop(columns=['gwb_code'], inplace=True)
    merged_df.sort_values(by='a_inw', ascending=False, inplace=True)
    
    return merged_df