import os
import numpy as np
import pandas as pd

from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from codebase.data.load_odin import make_ml_dataset, prepare_odin_stats, odin_add_buurtcode
from codebase.data.filters import filter_by_distance_and_duration, filter_by_origin, filter_by_destination, filter_by_motive, transport_modes
from codebase.data.column_names import transport_mode_col, id_col, punt_buurt_code_column
from codebase.plotting.plots import plot_confusion_matrix
from codebase.plotting.plots import plot_value_by_buurt_heatmap
from codebase.data.column_lists import (
    drop_cols, 
    numerical_cols,
    categorical_cols,
    ordinal_cols,
    binary_cols,
)


import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

class SktorchNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_layers=64):
        super(SktorchNN, self).__init__()
        layers = []
        for size in hidden_layers:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            input_dim = hidden_layers
        layers.append(nn.Linear(hidden_layers, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def create_sktorch_nn(
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_layers: list[int] = [64],
        max_epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 32,
        optimizer: str = 'adam',
        criterion: str = 'cross_entropy',
) -> NeuralNetClassifier:
    net = NeuralNetClassifier(
        SktorchNN(input_dim=input_dim, output_dim=output_dim, n_layers=n_layers, hidden_layers=hidden_layers),
        max_epochs=max_epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=optimizer,
        criterion=criterion,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    return net
    
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
        y_translation=None,
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
        y_translation=y_translation,
    )
    
    
    scaler = MinMaxScaler()
    model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, n_estimators=100, class_weight="balanced", verbose=verbose) if model is None else model
    pipeline = Pipeline([
        ('scaler', scaler),
        # ('tofloat32', ToFloat32()),
        ('model', model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # If y_translation is provided, reverse the mapping for y_test and y_pred
    if y_translation is not None:
        y_translation_reverse = {v: k for k, v in y_translation.items()}
        y_test = y_test.map(y_translation_reverse).values
        y_pred = pd.Series(y_pred, ).map(y_translation_reverse).values
    
    transport_modes_plot = {k: v for k, v in transport_modes.items() if k in np.unique(y_test)}
    classification_report_ = classification_report(y_test, y_pred, target_names=transport_modes_plot.values())
    print(classification_report_)
    accuracy = np.mean(y_pred == y_test)

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels=transport_modes_plot.values(), title=plot_title, savename=savename)

    return pipeline, (X_train, X_test, y_test, y_pred), accuracy 



def run_transferable_classification(
        odin_df: pd.DataFrame, 
        pipeline_transferable,
        cols_for_transferable, 
        necessary_columns,
        threshold_datapoints=100, 

        unused_target="a_inw",
        col_car_pred = "willingness_to_car_pred",
        col_cycle_pred = "willingness_to_cycle_pred",
        col_ebike_pred = "willingness_to_ebike_pred",
        col_walk_pred = "willingness_to_walk_pred",
        goal_value=7,
        motive_value=6,
        plot=True,
        df_save_path=None, 
        overwrite_existing=False,
    ):
    import os
    if df_save_path is None:
        df_save_path = f"data/classification_results/multiclass/transferable_classification_results_{goal_value}_{motive_value}_{threshold_datapoints}.csv"
    
    if not df_save_path.endswith(".csv"):
        df_save_path += ".csv"
    if os.path.exists(df_save_path) and not overwrite_existing:
        print(f"File {df_save_path} already exists. Loading existing results.")
        demographics_with_predictions = pd.read_csv(df_save_path)
        return demographics_with_predictions
    
    # Ensure the DataFrame has the necessary columns
    odin_df["BuurtCode"] = odin_df["WoPC"].astype(str)  
    odin_df[punt_buurt_code_column] = odin_df["BuurtCode"].astype(str)
    # Ensure the 'WoPC' column is present and format the 'BuurtCode' column for plotting later
    stats_df = prepare_odin_stats(odin_df, buurt_code_column="BuurtCode",)
    stats_df["WoPC"] = stats_df["BuurtCode"]
    stats_df = odin_add_buurtcode(stats_df, buurt_code_column="BuurtCode")
    stats_df[punt_buurt_code_column] = stats_df["BuurtCode"].apply(lambda x: "BU" + str(x)).astype(str)
    
    # Filter the DataFrame based on the threshold for the number of trips per BuurtCode 
    mask_count = stats_df["Count"] > threshold_datapoints
    demographics = stats_df[mask_count]
    # Set the goal and motive values
    demographics.loc[:, "Doel"] = goal_value         # Education
    demographics.loc[:, "KMotiefV"] = motive_value   # Education
    print(f"Contains {len(demographics)} rows with more than {threshold_datapoints} trips per BuurtCode.")
    demographics[unused_target] = 0
    # Make the categorical columns and remove the unused columns
    categorical_cols_for_transferable = [col for col in cols_for_transferable if col in categorical_cols]
    cols_to_drop_transferable = [col for col in demographics.columns if col not in cols_for_transferable + [unused_target, id_col]]
    # Turn the DataFrame into a machine learning dataset
    demographics_ml_X, _, _, _ = make_ml_dataset(
        demographics,
        target_col=unused_target,
        categorical_cols=categorical_cols_for_transferable,
        group_col=None,
        drop_cols=cols_to_drop_transferable,
        test_size=0.0001,
        ensure_common_labels=False
    )
    # Ensure the DataFrame has the necessary columns for the model
    demographics_ml_X = demographics_ml_X.dropna()
    missing_cols = set(necessary_columns) - set(demographics_ml_X.columns)
    for col in missing_cols:
        demographics_ml_X[col] = 0
    demographics_ml_X = demographics_ml_X[necessary_columns]

    # Predict the probabilities using the transferable model
    predicted_probs = pipeline_transferable.predict_proba(demographics_ml_X)
    # Create a DataFrame with the predictions
    demographics_ml_with_predictions = demographics_ml_X.copy()
    demographics_ml_with_predictions[transport_mode_col + "_pred"] = np.argmax(predicted_probs, axis=1)
    demographics_ml_with_predictions[col_car_pred] = predicted_probs[:, 0]  # Assuming index 0 corresponds to car
    demographics_ml_with_predictions[col_cycle_pred] = predicted_probs[:, 1]  # Assuming index 1 corresponds to cycling
    demographics_ml_with_predictions[col_ebike_pred] = predicted_probs[:, 2]  # Assuming index 2 corresponds to e-biking
    demographics_ml_with_predictions[col_walk_pred] = predicted_probs[:, 3]  # Assuming index 3 corresponds to walking

    demographics_with_predictions = demographics.copy()
    demographics_with_predictions = demographics_with_predictions.merge(
        demographics_ml_with_predictions[[transport_mode_col + "_pred", col_car_pred, col_cycle_pred, col_ebike_pred, col_walk_pred]],
        how="left",
        left_index=True,
        right_index=True
    )

    os.makedirs(os.path.dirname(df_save_path), exist_ok=True)
    demographics_with_predictions.to_csv(df_save_path, index=False)
    print(f"Saved predictions to {df_save_path}")
    
    if plot:
        for col in [col_car_pred, col_cycle_pred, col_ebike_pred, col_walk_pred]:
            plot_value_by_buurt_heatmap(
                demographics_with_predictions,
                col_name=col,
                # title=f"Willingness to {col.replace('willingness_to_', '').capitalize()} by Buurt according to Transferable Model",
                savename=f"graphics/classification_results/multiclass/transferable_{col}.png",
                show=plot
            )
    return demographics_with_predictions