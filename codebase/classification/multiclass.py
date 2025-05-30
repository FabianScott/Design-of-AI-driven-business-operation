import numpy as np
import pandas as pd

from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from codebase.data.load_odin import make_ml_dataset
from codebase.data.filters import filter_by_distance_and_duration, filter_by_origin, filter_by_destination, filter_by_motive, transport_modes
from codebase.data.column_names import transport_mode_col, id_col
from codebase.plotting.plots import plot_confusion_matrix
from codebase.data.column_lists import (
    drop_cols, 
    numerical_cols,
    categorical_cols,
    ordinal_cols,
    binary_cols
)


def run_multiclass_classification(
        df: pd.DataFrame,
        model: BaseEstimator = None, 
        test_size=0.02, 
        max_dist=np.inf, 
        origins=None, 
        destinations=None,
        location_level=0,
        motives=None,
        categorical_features=categorical_cols,
        drop_cols=drop_cols,
        plot=True, 
        savename=None,
        verbose=True,
        plot_title="Multiclass Classification",
        ) -> tuple:
    """

    Run a binary regression on the dataset to predict the probability of specific mode based on distance.
    The model is trained on a subset of the data where the distance is less than max_dist and the duration is not filtered.
    The model is then evaluated on a test set and the predicted probabilities are plotted against the actual values.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    model : BaseEstimator, optional
        The machine learning model to use. Default is None, which uses RandomForestClassifier.
    transport_mode : int, optional
        The transport mode to predict. Default is 5 (Bicycle).
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
        0: Buurt, 1: Gemeente, 2: Provincie, Default is 0.
    motives : list, optional
        A list of motives to filter the data. Default is None.
        If None, no filtering is applied.
    categorical_features : list, optional
        A list of categorical features to include in the model. Default is None.
    numerical_features : list, optional
        A list of numerical features to include in the model. Default is None.
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

    df_filtered = filter_by_origin(df, origins, level=location_level) if origins is not None else df
    df_filtered = filter_by_destination(df_filtered, destinations, level=location_level) if destinations else df_filtered
    df_filtered = filter_by_motive(df_filtered, motives) if motives else df_filtered
    try:
        df_filtered = filter_by_distance_and_duration(df_filtered, 0, max_dist, 0, np.inf)
    except KeyError as e:
        print(f"Skip filtering by distance and duration: {e}")
    
    X_train, X_test, y_train, y_test = make_ml_dataset(
        df_filtered,
        target_col=transport_mode_col,
        drop_cols=drop_cols,
        categorical_cols=categorical_features,
        test_size=test_size,
        group_col=id_col,
    )
    
    
    scaler = MinMaxScaler()
    model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, n_estimators=100, class_weight="balanced", verbose=verbose) if model is None else model
    pipeline = make_pipeline(scaler, model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if plot:
        transport_modes_plot = {k: v for k, v in transport_modes.items() if k in y_test.unique()}
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=transport_modes_plot.values(), title=plot_title, savename=savename)

        print(classification_report(y_test, y_pred, target_names=transport_modes_plot.values()))

    return pipeline, (X_train, X_test, y_test, y_pred)