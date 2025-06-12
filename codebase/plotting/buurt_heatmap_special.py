import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt

from codebase.data.column_names import punt_travel_time_column, punt_detour_column, willingness_to_cycle_column, punt_buurt_code_column
from codebase.buurt_calculations.buurt_calculations import willingness_to_cycle
from codebase.data.codebook_dicts import transport_modes
from codebase.data.filters import filter_by_time
from codebase.data.load_buurt import load_buurt_data

def plot_willingness_by_buurt_heatmap(
        punt2, 
        mode, 
        location, 
        show=True, 
        savename=None, 
        cmap='viridis', 
        willingness_function=willingness_to_cycle,
        multiply_by_population=False,
        transport_mode_str=None,
        pipeline=None,
        max_val=100,
        min_val=0,
        title_vehicle=None,
        ):
    df_punt = load_buurt_data(punt2, mode=mode,)
    gdf = gpd.read_file("data/WijkBuurtkaart_2023_v2/wijkenbuurten_2023_v2.gpkg", layer="buurten")
    # Remove the empty buurts
    gdf = gdf[gdf["aantal_inwoners"] > 0]
    # Use only the closest location to each buurt:
    df_punt = filter_by_time(df_punt, np.inf) 
    df_punt["buurtcode"] = df_punt["bu_code"]
    # Add the willingness to cycle % column to the dataframe
    willingness_column_name = f"{punt2}_by_{mode}_willingness"
    df_punt[willingness_column_name] = willingness_function(df_punt[punt_travel_time_column].values, 
                                                            mode=mode, 
                                                            location=location,
                                                            pipeline=pipeline,
                                                            ) * 100
    
    # Merge the two dataframes on the buurtcode, and fill the NaN values with 0
    gdf = gdf.merge(df_punt, on='buurtcode', how='left')
    # gdf = gdf.fillna(0)
    if multiply_by_population:
        # Multiply the willingness by the population of the buurt, divided by 100 because the willingness is in percentage
        gdf[willingness_column_name] *= gdf["aantal_inwoners"].values / 100
    
    fig = plt.figure(figsize=(10, 10), frameon=False)
    gdf.plot(column=willingness_column_name, cmap=cmap, markersize=5, legend=True, vmin=min_val, vmax=max_val)
    title_vehicle = "cycle" if title_vehicle is None else title_vehicle
    plt.title(f"Heatmap of willingness to {transport_mode_str} to {punt2} in %")
    plt.axis("off")
    plt.tight_layout()
    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, bbox_inches='tight', dpi=300)
    if show:
        plt.show()