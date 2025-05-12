import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt

from codebase.data.column_names import punt_travel_time_column, punt_detour_column, willingness_to_cycle_column
from codebase.buurt_calculations.buurt_calculations import filter_by_time, willingness_to_cycle
from codebase.data.codebook_dicts import transport_modes
from codebase.data.load_buurt import load_buurt_data

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