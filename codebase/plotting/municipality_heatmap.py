import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from codebase.data_manipulation.column_names import municipality_name_column, geofile_population_column
from codebase.buurt_calculations.municipality_calculations import load_municipality_geometry

def plot_detour_map(
    det: pd.DataFrame,
    category: str,
    modes: tuple[str, ...],
    gpkg: Path = Path("data/wijkenbuurten_2023_v2.gpkg"),
    population_threshold: int = 0,
    missing_kwds: dict = None
) -> None:
    """
    Plot a map of the population-weighted detour factor for a given category and modes.
    """
    gdf = load_municipality_geometry(gpkg).merge(det, on=municipality_name_column, how="left")
    # Filter out municipalities with population below the threshold
    gdf["plot_col"] = gdf[gdf[geofile_population_column] >= population_threshold]["pop_weighted_detour"]

    if missing_kwds is None:
        missing_kwds = {"color": "lightgrey", "label": f"pop ≤ {population_threshold:,}"}
    
    ax = gdf.plot(
        column="plot_col",
        cmap="OrRd",
        linewidth=0.2,
        edgecolor="black",
        figsize=(8, 10),
        legend=True,
        #make legend smaller
        legend_kwds={"label": "Population-weighted detour factor", "shrink": 0.5},
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(
        f"Population-weighted detour factor to '{category}' "
        f"destinations ({', '.join(modes)})" + f" for municipalities with population > {population_threshold:,}" if population_threshold > 0 else "",
        fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"graphics/detour_map_{category}_{'_'.join(modes)}_{population_threshold}.png",
        dpi=300, bbox_inches="tight")
    plt.show()
