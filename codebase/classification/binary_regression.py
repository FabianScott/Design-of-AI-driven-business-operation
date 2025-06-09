import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from codebase.data.load_odin import make_ml_dataset
from codebase.data.filters import filter_by_distance_and_duration, filter_by_origin, filter_by_destination, filter_by_motive, transport_modes
from codebase.data.column_names import transport_mode_col, distance_col, id_col
from codebase.plotting.plots import plot_binary_regression


def run_binary_regression(
        df: pd.DataFrame, 
        transport_modes_predict=[5,], 
        test_size=0.02, 
        max_dist=np.inf, 
        origins=None, 
        destinations=None,
        location_level=0,
        motives=None,
        additional_features=None,
        plot=True, 
        savename=None
        ) -> tuple:
    """

    Run a binary regression on the dataset to predict the probability of specific mode based on distance.
    The model is trained on a subset of the data where the distance is less than max_dist and the duration is not filtered.
    The model is then evaluated on a test set and the predicted probabilities are plotted against the actual values.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    transport_modes : list[int], optional
        The transport modes to predict. Default is 5 (Bicycle).
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.02.
    max_dist : int, optional
        The maximum distance (in 100m) to filter the data. Default is 500.
    origins : list, optional
        A list of origins to filter the data. Default is None.
        If None, no filtering is applied.
    destinations : list, optional
        A list of destinations to filter the data. Default is None.
        If None, no filtering is applied.
    location_level : int, optional
        0: Buurt, 1: Gemeente, 2: Provincie. Default is 0.
    motives : list, optional
        A list of motives to filter the data. Default is None.
        If None, no filtering is applied.
    additional_features : list, optional
        A list of additional features to include in the model. Default is None.
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
    df_filtered = filter_by_origin(df_filtered, origins, level=location_level) if origins else df_filtered
    df_filtered = filter_by_destination(df_filtered, destinations, level=location_level) if destinations else df_filtered
    df_filtered = filter_by_motive(df_filtered, motives) if motives else df_filtered

    X_train, X_test, y_train, y_test = make_ml_dataset(
        df_filtered,
        target_col=transport_mode_col,
        target_vals=transport_modes_predict,  # see load_data/filters.py for KHvm value dictionary
        drop_cols=[col for col in df.columns if col not in [transport_mode_col, distance_col, id_col] + 
                   (additional_features if additional_features is not None else [])],
        categorical_cols=None,
        test_size=test_size,
        group_col=id_col,
        )

    scaler = MinMaxScaler()
    model = LogisticRegression()
    pipeline = make_pipeline(scaler, model)
    try:
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error fitting the model: {e}")
        print("Check if the target variable has only one class in the training set.")
        return None, None

    y_pred = pipeline.predict_proba(X_test)[:, 1]  # Get the probability of cycling

    if plot:
        plot_binary_regression(X_test, y_test, y_pred, transport_modes_predict, motives, savename=savename)

    return pipeline, (X_train, X_test, y_test, y_pred)

def binary_pipeline_as_willingness_function(time_array, pipeline, **kwargs):
    """
    Returns the willingness to pay for a given time array.
    """
    # Convert time_array to DataFrame with appropriate feature names
    df_time = pd.DataFrame(time_array, columns=pipeline.feature_names_in_)
    
    # Predict probabilities using the pipeline
    probabilities = pipeline.predict_proba(df_time)[:, 1]
    
    return probabilities