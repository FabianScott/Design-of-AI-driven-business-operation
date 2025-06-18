import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from codebase.data_manipulation.column_names import (
    municipality_name_column,
    demographics_population_column,
    geofile_population_column 

)
from codebase.data_manipulation.filters import filter_by_time

def load_municipality_geometry(gpkg_path: Path = Path("data\WijkBuurtkaart_2023_v2\wijkenbuurten_2023_v2.gpkg")) -> gpd.GeoDataFrame:
    """Return GeoDataFrame with ['municipality_name', 'geometry'].""" 
    gdf = gpd.read_file(gpkg_path, layer="gemeenten")
    if "gemeentenaam" not in gdf.columns:
        raise ValueError("'gemeentenaam' column not found")
    gdf = (
        gdf[["gemeentenaam", "geometry"]]
        .rename(columns={"gemeentenaam": municipality_name_column})
        .assign(municipality_name=lambda d: d[municipality_name_column].str.title())
    )
    return gdf


def population_weighted_detour(trips: pd.DataFrame,
                               demographics: pd.DataFrame) -> pd.DataFrame:
    """DataFrame ['municipality_name', 'pop_weighted_detour'].""" 
    merged = (
        trips.merge(
            demographics[["neighbourhood_code",
                          municipality_name_column,
                          "population"]],
            left_on="origin_neighbourhood",
            right_on="neighbourhood_code",
            how="left",
            validate="many_to_one"
        ).dropna(subset=["population"])
    )

    result = (
        merged.groupby(municipality_name_column, as_index=False)
              .apply(lambda df: (df["detour_factor"] * df["population"]).sum()
                                 / df["population"].sum())
              .rename(columns={0: "pop_weighted_detour"})
    )
    result[municipality_name_column] = result[municipality_name_column].str.title()
    return result

def weighted_detour_by_municipality(trips: pd.DataFrame,
                                    demo:  pd.DataFrame,
                                    ) -> pd.DataFrame:
    """
    Population-weighted mean detour factor per municipality.
    Returns DataFrame [municipality_name_column, 'pop_weighted_detour'].
    """
    trips_filtered = filter_by_time(trips, max_time=np.inf)
    merged = (
        trips_filtered.merge(
            demo[["gwb_code", municipality_name_column, demographics_population_column]],
            left_on="bu_code", right_on="gwb_code",
            how="left", validate="many_to_one")
          .dropna(subset=[demographics_population_column])
    )

    w = (
    merged.groupby(municipality_name_column)
          .apply(lambda x: pd.Series({
              "pop_weighted_detour": (x["omrijdfactor"] * x[demographics_population_column]).sum()
                                     / x[demographics_population_column].sum(),
              geofile_population_column: x[demographics_population_column].sum()
          }))
          .reset_index()
)
    w[municipality_name_column] = w[municipality_name_column].str.title()
    return w

def detour_vs_population(trips: pd.DataFrame,
                         demographics: pd.DataFrame) -> pd.DataFrame:
    """DataFrame ['municipality_name', 'population', 'pop_weighted_detour']."""
    detour_df = population_weighted_detour(trips, demographics)
    pop_df = (
        demographics.groupby(municipality_name_column, as_index=False)["population"]
                    .sum()
    )
    return detour_df.merge(pop_df, on=municipality_name_column,
                           how="left").dropna()
