import numpy as np
import pandas as pd
from codebase.data_manipulation.filters import (
    filter_by_distance_and_duration,
    filter_by_origin,
    filter_by_destination,
    filter_by_motive
)
from codebase.data_manipulation.column_names import (
    transport_mode_col,
    distance_col,
    id_col
)
from codebase.data_manipulation.column_lists import (
    drop_cols
)
from codebase.data_loading.load_odin import load_odin


def make_ml_dataset(
        df: pd.DataFrame,
        target_col,
        drop_cols,
        categorical_cols=None,
        target_vals=None,
        test_size=0.2,
        random_state=42,
        stratification_col=None,
        group_col=None,
        y_translation: dict=None,
        ensure_common_labels: bool = True,
        dropna=False,
        ) -> tuple:
    """
    Splits the dataset into training and testing sets.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GroupShuffleSplit

    if stratification_col is not None and group_col is not None:
        raise ValueError("Cannot use both stratification_col and group_col for splitting.")

    # Drop specified columns
    drop_cols_to_use = [col for col in drop_cols if (col in df.columns) and not col == group_col] if drop_cols else []
    df_: pd.DataFrame = df.drop(columns=drop_cols_to_use, errors='ignore')

    # Check for NaN values in the dataframe
    if dropna and df_.isnull().values.any():
        print(f"Dataframe contains NaN values in columns: {df_.columns[df_.isnull().any()].tolist()}")
        print("Warning: These will be dropped.")
        df_ = df_.dropna(axis=0)

    # Split the data into features and target
    X = df_.drop(columns=[target_col])
    categorical_cols_to_use = [col for col in categorical_cols if col in X.columns] if categorical_cols else []
    X = pd.get_dummies(X, columns=categorical_cols_to_use, drop_first=True, dtype=np.int64)
    # Cast all non categorical columns to float
    X = X.astype({col: float for col in X.columns if col.split("_")[0] not in categorical_cols_to_use})
    y: pd.DataFrame = df_[target_col].isin(target_vals) if target_vals is not None else df_[target_col]
    y = y.astype(np.int64)
    if y_translation:
        y = y.map(y_translation).fillna(0).astype(np.int64)
    stratification = df_[stratification_col] if stratification_col else None

    # Split the data into training and testing sets
    if group_col:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df_[group_col]))
        X.drop(columns=[group_col], inplace=True, errors='ignore')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratification)

    if ensure_common_labels:
        # 1. Find the set of labels that occur in both y_train and y_test
        common_labels = np.intersect1d(np.unique(y_train), np.unique(y_test))

        print(f"Common labels: {common_labels}")

        # 2. Create boolean masks
        train_mask = y_train.isin(common_labels)
        test_mask = y_test.isin(common_labels)

        # 3. Filter both sets
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    return X_train, X_test, y_train, y_test

def load_odin_as_ml_dataset(
        transport_modes_predict: list = [5], # Bicycle
        years: list = None,
        odin_df: pd.DataFrame = None,
        test_size: float = 0.2,
        max_dist: float = np.inf,
        origins: list = None,
        destinations: list = None,
        motives: list = None,
        location_level: int = 0,
        drop_cols: list = drop_cols,
):
    
    df = load_odin(years=years, do_apply_ignore_rules=True, dropna=False) if odin_df is None else odin_df
    df_filtered = filter_by_distance_and_duration(df, 0, max_dist, 0, np.inf)
    df_filtered = filter_by_origin(df_filtered, origins, level=location_level) if origins else df_filtered
    df_filtered = filter_by_destination(df_filtered, destinations, level=location_level) if destinations else df_filtered
    df_filtered = filter_by_motive(df_filtered, motives) if motives else df_filtered

    X_train, X_test, y_train, y_test = make_ml_dataset(
        df_filtered,
        target_col=transport_mode_col,
        target_vals=transport_modes_predict,  # see load_data/filters.py for KHvm value dictionary
        drop_cols=drop_cols,
        categorical_cols=None,
        test_size=test_size,
        group_col=id_col,
        )
    
    return X_train, X_test, y_train, y_test