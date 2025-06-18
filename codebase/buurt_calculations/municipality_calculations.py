from pathlib import Path
import geopandas as gpd
import pandas as pd

def load_municipality_geometry(gpkg_path: Path) -> gpd.GeoDataFrame:
    """Return GeoDataFrame with ['municipality_name', 'geometry'].""" 
    gdf = gpd.read_file(gpkg_path, layer="gemeenten")
    if "gemeentenaam" not in gdf.columns:
        raise ValueError("'gemeentenaam' column not found")
    gdf = (
        gdf[["gemeentenaam", "geometry"]]
        .rename(columns={"gemeentenaam": "municipality_name"})
        .assign(municipality_name=lambda d: d["municipality_name"].str.title())
    )
    return gdf


def population_weighted_detour(trips: pd.DataFrame,
                               demographics: pd.DataFrame) -> pd.DataFrame:
    """DataFrame ['municipality_name', 'pop_weighted_detour'].""" 
    merged = (
        trips.merge(
            demographics[["neighbourhood_code",
                          "municipality_name",
                          "population"]],
            left_on="origin_neighbourhood",
            right_on="neighbourhood_code",
            how="left",
            validate="many_to_one"
        ).dropna(subset=["population"])
    )

    result = (
        merged.groupby("municipality_name", as_index=False)
              .apply(lambda df: (df["detour_factor"] * df["population"]).sum()
                                 / df["population"].sum())
              .rename(columns={0: "pop_weighted_detour"})
    )
    result["municipality_name"] = result["municipality_name"].str.title()
    return result

def weighted_detour_by_municipality(trips: pd.DataFrame,
                                    demo:  pd.DataFrame) -> pd.DataFrame:
    """
    Population-weighted mean detour factor per municipality.
    Returns DataFrame ['gm_naam', 'pop_weighted_detour'].
    """
    merged = (
        trips.merge(
            demo[["gwb_code", "gm_naam", "a_inw"]],
            left_on="bu_code", right_on="gwb_code",
            how="left", validate="many_to_one")
          .dropna(subset=["a_inw"])
    )

    w = (
        merged.groupby("gm_naam")
              .apply(lambda x: (x["omrijdfactor"] * x["a_inw"]).sum()
                               / x["a_inw"].sum())
              .reset_index(name="pop_weighted_detour")
    )
    w["gm_naam"] = w["gm_naam"].str.title()
    return w

def detour_vs_population(trips: pd.DataFrame,
                         demographics: pd.DataFrame) -> pd.DataFrame:
    """DataFrame ['municipality_name', 'population', 'pop_weighted_detour']."""
    detour_df = population_weighted_detour(trips, demographics)
    pop_df = (
        demographics.groupby("municipality_name", as_index=False)["population"]
                    .sum()
    )
    return detour_df.merge(pop_df, on="municipality_name",
                           how="left").dropna()
