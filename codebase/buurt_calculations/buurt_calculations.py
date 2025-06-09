import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codebase.data.load_demographics import load_demograhics
from .willingness import willingness_to_cycle
from codebase.data.load_buurt import load_buurt_data
from codebase.data.column_names import (
    demographics_population_column,
    demographics_buurt_code_column,
    punt_travel_time_column,
    punt_buurt_code_column,
    willingness_to_cycle_column,
    punt_detour_column,
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

def align_by_buurt(df_filtered: pd.DataFrame, df_demographics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    df_filtered.drop_duplicates(subset=[punt_buurt_code_column], keep='first', inplace=True)
    df_location.drop_duplicates(subset=[demographics_buurt_code_column], keep='first', inplace=True)

    return df_filtered, df_location

def get_total_willingness_to_cycle_in_buurts(df: pd.DataFrame, location, within_mins, mode, df_demographics=None, ):
    df_demographics = load_demograhics() if df_demographics is None else df_demographics
    df_filtered = filter_by_time(df, within_mins)
    df_filtered, df_location = align_by_buurt(df_filtered, df_demographics)
    # Get the unique buurt codes from the filtered dataframe
    df_filtered = add_willingness_to_cycle_column(df_filtered, location, mode=mode)
    # Filter the demographics dataframe to only include the relevant buurt codes
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
    nl_total = df_demographics[demographics_population_column][0]
    df_punt = load_buurt_data(punt1, mode=mode)
    df_punt, df_demographics = align_by_buurt(df_punt, df_demographics)
    
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
        demographics[['gwb_code', 'a_inw', 'gm_naam']], 
        how='left',
        left_on='bu_code',
        right_on='gwb_code'
    )

    merged_df.drop(columns=['gwb_code'], inplace=True)
    merged_df.sort_values(by='a_inw', ascending=False, inplace=True)
    
    return merged_df

def calculate_added_willingness(
        df_punt, 
        df_demographics,
        mode="fiets",
        location="Education",
        detour_max=1.2,
        detour_reduction=None,
        improvement_column="n_extra_inhabitants_now_willing",
        savename=None,
        plot=True,
        col_to_plot="n_extra_inhabitants_now_willing",
        age_group_column=None,
        ):
    """
    Calculate the added willingness to cycle given a maximum detour factor.
    If detour reduction is not None, it will be used to reduce the travel time as 
    a proportion of the original travel time applied only to buurts over detour_max.
    If detour reduction is None, the reduction will be calculated based on the 
    maximum detour factor.
    The new travel time will be stored in a new column with the suffix "_new".
    """
    # Remove duplicates and keep only the smallest travel time
    df_filtered = filter_by_time(df_punt, max_time=np.inf)
    # Align the dataframes by buurt code
    df_filtered, df_location = align_by_buurt(df_filtered, df_demographics,)

    new_travel_time_col = punt_travel_time_column + "_new"
    new_willingness_col = willingness_to_cycle_column + "_new"

    detour_is_greater = df_filtered[punt_detour_column] > detour_max
    new_time = df_filtered[punt_travel_time_column].copy()
    # Calculate the new travel time based on the detour factor
    
    if detour_reduction is not None:
        new_time[detour_is_greater] = df_filtered[punt_travel_time_column][detour_is_greater] * detour_reduction
    else:
        # Calculate the improvement ratio based on the maximum detour factor
        improvement_ratio = detour_max / df_filtered[punt_detour_column][detour_is_greater]
        new_time[detour_is_greater] = df_filtered[punt_travel_time_column][detour_is_greater] * improvement_ratio
    
    df_filtered[new_travel_time_col] = new_time

    df_filtered = add_willingness_to_cycle_column(df_filtered, location=location, mode=mode, travel_time_col=punt_travel_time_column, willingness_col=willingness_to_cycle_column)
    df_filtered = add_willingness_to_cycle_column(df_filtered, location=location, mode=mode, travel_time_col=new_travel_time_col, willingness_col=new_willingness_col)
    
    # Calculate the difference in willingness to cycle
    df_filtered["willingness_diff"] = df_filtered[new_willingness_col] - df_filtered[willingness_to_cycle_column]
    # Calculate the proportion of the difference relative to the original willingness to cycle
    df_filtered["willingness_diff_proportion"] = (df_filtered["willingness_diff"] / df_filtered[willingness_to_cycle_column])
    
    improvement_mask = df_filtered["willingness_diff"] > 0
    df_filtered[improvement_column] = 0
    demo_col_to_use = demographics_population_column if age_group_column is None else age_group_column
    n_improvement = df_location[demo_col_to_use][improvement_mask].values * df_filtered["willingness_diff"][improvement_mask].values
    df_filtered.loc[improvement_mask, improvement_column] = np.round(n_improvement)
    
    if plot:
        from codebase.plotting.plots import plot_value_by_buurt_heatmap
        plot_value_by_buurt_heatmap(
            df_filtered, 
            col_name=col_to_plot, 
            show=True, 
            savename=savename, 
            cmap='viridis'
        )
        plt.savefig('ExtraInhabits', dpi=300, bbox_inches='tight')
    return df_filtered


def read_all_punt_to_punt(punten, modes):
    """
    Reads all punt to punt files defined in the list of punt names and modes.
    """
    matrix_data = {}

    for punt in punten:
        for mode in modes:
            key = f"{punt}_{mode}"
            try:
                df = load_buurt_data(punt, mode)
                matrix_data[key] = df.copy()
            except Exception as e:
                print(f"Skipping {key}: {e}")

    return matrix_data

def make_detour_matrix(matrix_data, savename=None):
    """
    Makes a matrix of the read all punt to punt dataframes. Takes matrix_data as input.
    """ 
    # Define bin edges
    bins = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, float("inf")]

    # Generate labels automatically
    labels = [
        f"{bins[i]:.2f}–{bins[i+1]:.2f}" if bins[i+1] != float("inf") else f">{bins[i]:.2f}"
        for i in range(len(bins) - 1)
    ]

    proportion_matrix = {}
    for key, df in matrix_data.items():
        # Bin the omrijdfactor
        df["bin"] = pd.cut(df["omrijdfactor"], bins=bins, labels=labels, right=False)
        
        # Count per bin
        counts = df["bin"].value_counts().reindex(labels, fill_value=0)
        total = counts.sum()
        
        # Store row-wise proportions
        proportions = counts / total if total > 0 else [0] * len(labels)
        proportion_matrix[key] = proportions


    # Convert to matrix DataFrame
    matrix_df = pd.DataFrame(proportion_matrix).T  # Transpose so rows are punt_mode
    matrix_df.index.name = "punt_mode"
    matrix_df.columns.name = "omrijdfactor_bin"

    # Display as a table
    #print(matrix_df)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix_df, annot=True, cmap='Blues', linewidths=0.5, cbar_kws={'label': 'Percentage'})
    plt.title("Detour Factor Distribution form Punt to Buurt and Mode")
    plt.ylabel("Punt to Buurt and Mode")
    plt.xlabel("Detour Factor Bin")
    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches='tight', dpi=300)
    plt.show()
