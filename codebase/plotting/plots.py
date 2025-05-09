import os
import numpy as np
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from codebase.load_data import load_buurt_data
from codebase.buurt_calculations import filter_by_time, willingness_to_cycle, punt_travel_time_column
from codebase.load_data.column_names import distance_col


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues', show=True, save_path=None):
    """
    Plots a confusion matrix using matplotlib and seaborn.

    Parameters:
    cm (array-like): Confusion matrix data.
    labels (list): List of labels for the confusion matrix.
    title (str): Title of the plot.
    cmap (str): Colormap to use for the heatmap.

    Returns:
    None
    """


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()


def plot_willingness_by_buurt_heatmap(punt2, mode, location, show=True, savename=None, cmap='viridis'):
    df_punt = load_buurt_data(punt2, mode=mode,)
    gdf = gpd.read_file("data/WijkBuurtkaart_2023_v2/wijkenbuurten_2023_v2.gpkg", layer="buurten")
    # Remove the empty buurts
    gdf = gdf[gdf["aantal_inwoners"] > 0]
    # Use only the closest location to each buurt:
    df_punt = filter_by_time(df_punt, np.inf) 
    df_punt["buurtcode"] = df_punt["bu_code"]
    # Add the willingness to cycle % column to the dataframe
    willingness_column_name = f"{punt2}_by_{mode}_willingness"
    df_punt[willingness_column_name] = willingness_to_cycle(df_punt[punt_travel_time_column].values, mode=mode, location=location) # gdf["aantal_inwoners"]
    # Merge the two dataframes on the buurtcode, and fill the NaN values with 0
    gdf = gdf.merge(df_punt, on='buurtcode', how='left')
    gdf = gdf.fillna(0)

    fig = plt.figure(figsize=(10, 10), frameon=False)
    gdf.plot(column=willingness_column_name, cmap=cmap, markersize=5, legend=True)

    plt.title(f"Heatmap of willingness to cycle to {punt2} by {mode} in %")
    plt.axis("off")
    plt.tight_layout()
    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, bbox_inches='tight', dpi=300)
    if show:
        plt.show()

def plot_binary_regression(X_test, y_test, y_pred, transport_mode_str, savename=None):
    # Bin settings
    bins = 50

    # Compute average actual cycling per bin
    bin_means, bin_edges, _ = binned_statistic(X_test[distance_col].values.flatten(), y_test.values.flatten(), statistic='mean', bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Plot predicted and actual
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[distance_col], y_pred, label="Predicted probability", alpha=0.5, color="orange", s=10)
    plt.scatter(X_test[distance_col], y_test, label="Actual binary value", alpha=0.5, color="blue", s=10)
    plt.plot(bin_centers, bin_means, label=f"Actual {transport_mode_str} rate (binned)", color="green", linewidth=2)

    # add the histogram of the actual values
    plt.xlabel("Distance (100m)")
    plt.ylabel(f"Predicted probability of {transport_mode_str}")
    plt.title(f"Predicted probability of {transport_mode_str} by distance")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if savename:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=300)
    plt.show()