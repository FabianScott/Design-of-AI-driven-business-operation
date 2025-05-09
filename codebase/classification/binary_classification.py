import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import binned_statistic

from codebase.load_data.load_odin import make_ml_dataset
from codebase.load_data.filters import filter_by_distance_and_duration, filter_by_origin, transport_modes


def run_binary_regression(df: pd.DataFrame, transport_mode=5, test_size=0.02, max_dist=np.inf, origins=None, destinations=None, plot=True, savename=None) -> tuple:
    """

    Run a binary regression on the dataset to predict the probability of specific mode based on distance.
    The model is trained on a subset of the data where the distance is less than max_dist and the duration is not filtered.
    The model is then evaluated on a test set and the predicted probabilities are plotted against the actual values.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    transport_mode : int, optional
        The transport mode to predict. Default is 5 (Bicycle).
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.02.
    max_dist : int, optional
        The maximum distance (in 100m) to filter the data. Default is 500.
    origins : list, optional
        A list of origins to filter the data. Default is None.
        If None, no filtering is applied.
    plot : bool, optional
        Whether to plot the predicted probabilities against the actual values. Default is True.
    savename : str, optional
        The name of the file to save the plot. If None, the plot will not be saved. Default is None.

    Returns
    -------
    tuple
        A tuple containing the trained model and the test set data (X_test, y_test, y_pred).
        X_train : pd.DataFrame
            The training set features.
        X_test : pd.DataFrame
            The test set features.
        y_test : pd.Series
            The actual values for the test set.
        y_pred : np.ndarray
            The predicted probabilities for the test set.
    """
    df_filtered = filter_by_distance_and_duration(df, 0, max_dist, 0, np.inf)
    df_filtered = filter_by_origin(df_filtered, origins) if origins else df_filtered
    df_filtered = filter_by_origin(df_filtered, destinations) if destinations else df_filtered

    X_train, X_test, y_train, y_test = make_ml_dataset(
        df_filtered,
        target_col="KHvm",
        target_val=transport_mode,  # see load_data/filters.py for KHvm value dictionary
        drop_cols=[col for col in df.columns if col not in ["KHvm", "AfstV"]],
        categorical_cols=None,
        test_size=test_size
        )

    scaler = MinMaxScaler()
    model = LogisticRegression()
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict_proba(X_test)[:, 1]  # Get the probability of cycling

    if plot:
        # Bin settings
        bins = 50

        # Compute average actual cycling per bin
        bin_means, bin_edges, _ = binned_statistic(X_test.values.flatten(), y_test.values.flatten(), statistic='mean', bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Plot predicted and actual
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_pred, label="Predicted probability", alpha=0.5, color="orange", s=10)
        plt.scatter(X_test, y_test, label="Actual binary value", alpha=0.5, color="blue", s=10)
        plt.plot(bin_centers, bin_means, label=f"Actual {transport_modes[transport_mode]} rate (binned)", color="green", linewidth=2)

        # add the histogram of the actual values
        plt.xlabel("Distance (100m)")
        plt.ylabel(f"Predicted probability of {transport_modes[transport_mode]}")
        plt.title(f"Predicted probability of {transport_modes[transport_mode]} by distance")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if savename:
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename, dpi=300)
        plt.show()

    return pipeline, (X_train, X_test, y_test, y_pred)