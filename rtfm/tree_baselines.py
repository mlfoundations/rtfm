import json
from typing import Dict, Any, Tuple, Literal

import numpy as np
import pandas as pd
import tableshift
import torch
from ray import tune
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from tableshift.core.features import is_categorical
from tabpfn import TabPFNClassifier
from tune_sklearn import TuneSearchCV
from xgboost import XGBClassifier


def bool_cols_to_categorical(df: pd.DataFrame):
    """Convert bool columns to integer.

    XGBoost treats bool dtype as object dtype (which is not supported),
    so these need to be handled separately."""
    for c in df.columns:
        if df[c].dtype.str == "|b1":
            df[c] = df[c].astype(int)
    return df


# Matches https://arxiv.org/pdf/2106.11959.pdf; see Table 16
XGB_HPARAM_GRID = {
    "learning_rate": tune.loguniform(1e-5, 1.0),
    "max_depth": tune.randint(3, 10),
    "min_child_weight": tune.loguniform(1e-8, 1e5),
    "subsample": tune.uniform(0.5, 1),
    "colsample_bytree": tune.uniform(0.5, 1),
    "colsample_bylevel": tune.uniform(0.5, 1),
    "gamma": tune.loguniform(1e-8, 1e2),
    "lambda": tune.loguniform(1e-8, 1e2),
    "alpha": tune.loguniform(1e-8, 1e2),
    "max_bin": tune.choice([128, 256, 512]),
}


def tune_xgb(X_tr: pd.DataFrame, y_tr: pd.Series, n_trials, num_classes: int = 2):
    assert num_classes >= 2
    if num_classes == 2:
        clf = XGBClassifier(
            enable_categorical=True, tree_method="hist", objective="binary:logistic"
        )
    else:  # multiclass
        clf = XGBClassifier(
            enable_categorical=True,
            tree_method="hist",
            objective="multi:softmax",
            num_class=num_classes,
        )

    cv = min(len(X_tr), 3)
    if len(X_tr) > 1 and (all(y_tr.value_counts() > cv)) and n_trials > 1:
        tune_search = TuneSearchCV(
            clf,
            XGB_HPARAM_GRID,
            n_trials=n_trials,
            early_stopping=False,
            search_optimization="hyperopt",
            cv=cv,
        )
        return tune_search.fit(X_tr, y_tr)
    elif n_trials == 1:
        print(
            "[WARNING n_trials is 1; not hyperparameter tuning / using default hparams."
        )
        return clf.fit(X_tr, y_tr)
    else:
        print(
            "WARNING: cannot tune hparams with dataset of length 1 or only 1 class per CV fold; "
            "using default params."
        )
        return clf.fit(X_tr, y_tr)


def tune_logistic_regression(X_tr, y_tr, num_trials):
    assert num_trials > 1, f"logistic regression without tuning is not implemented."

    # Define the logistic regression model
    log_reg = LogisticRegression(max_iter=10000)

    cv = min(len(X_tr), 3)
    if len(X_tr) > 1 and (all(y_tr.value_counts() > cv)) and num_trials > 1:
        # Define a range of values for C (regularization parameter)
        C_values = np.logspace(-4, 4, num_trials - 1).tolist() + [
            1e10
        ]  # Adding a very large value to represent "no regularization"

        # Create the parameter grid
        param_grid = {"C": C_values}

        # Create the GridSearchCV object
        grid_search = GridSearchCV(
            log_reg, param_grid, cv=cv, scoring="accuracy", n_jobs=-1
        )

        # Perform the grid search
        return grid_search.fit(X_tr, y_tr)

    elif num_trials == 1:
        print(
            "[WARNING n_trials is 1; not hyperparameter tuning / using default hparams."
        )
        return log_reg.fit(X_tr, y_tr)
    else:
        print(
            "WARNING: cannot tune hparams with dataset of length 1 or only 1 class per CV fold; "
            "using default params."
        )
        return log_reg.fit(X_tr, y_tr)


def tune_tabpfn(X_tr, y_tr, n_trials, subsample_features=True):
    clf = TabPFNClassifier(
        device="cpu" if not torch.cuda.is_available() else "cuda",
        subsample_features=subsample_features,
        batch_size_inference=4,
    )
    cv = min(len(X_tr), 3)
    max_n_ensemble = 2 * min(X_tr.shape[1], 100)
    grid_values = np.arange(3, max_n_ensemble)

    if len(X_tr) > 1 and (all(y_tr.value_counts() > cv)) and len(grid_values):
        if len(grid_values) < n_trials:
            tune_search = GridSearchCV(
                clf,
                param_grid={"N_ensemble_configurations": grid_values},
                scoring="accuracy",
                cv=cv,
                verbose=2,
                error_score="raise",
            )
        elif len(grid_values):
            tune_search = RandomizedSearchCV(
                clf,
                param_distributions={"N_ensemble_configurations": grid_values},
                n_iter=n_trials,
                scoring="accuracy",
                cv=cv,
                verbose=2,
                error_score="raise",
            )

        return tune_search.fit(X_tr, y_tr)
    else:
        return clf.fit(X_tr, y_tr)


def train_tune_and_compute_metrics(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    n_trials,
    model_type: Literal["xgboost", "tabpfn", "logistic_regression"],
    random_seed: int = None,
) -> Tuple[Any, Dict[str, Any]]:
    if random_seed is not None:
        np.random.seed(random_seed)

    num_classes = len(set(y_tr.unique().tolist() + y_te.unique().tolist()))
    if model_type == "xgboost":
        clf = tune_xgb(X_tr, y_tr, n_trials, num_classes=num_classes)
    elif model_type == "tabpfn":
        clf = tune_tabpfn(X_tr, y_tr, n_trials)
    elif model_type == "logistic_regression":
        clf = tune_logistic_regression(X_tr, y_tr, num_trials=n_trials)

    preds = clf.predict(X_te)
    acc = accuracy_score(preds, y_te)

    majority_acc = y_te.value_counts().max() / len(y_te)

    return clf, dict(
        acc=acc,
        majority_acc=majority_acc,
        n_test=len(y_te),
        n_train=len(y_tr),
        n_trials=n_trials,
        y_test_frac=json.dumps((y_te.value_counts() / len(y_te)).to_dict()),
        y_train_frac=json.dumps((y_tr.value_counts() / len(y_tr)).to_dict()),
    )


def label_encode_column(
    train_ser: pd.Series, test_ser: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    le = LabelEncoder()
    # Convert the column to string type to handle mixed types
    combined_data = pd.concat([train_ser.astype(str), test_ser.astype(str)], axis=0)
    le.fit(combined_data)
    # Transform both train and test data
    train_ser = le.transform(train_ser.astype(str))
    test_ser = le.transform(test_ser.astype(str))
    return pd.Series(train_ser), pd.Series(test_ser)


def label_encode_train_test(train_df, test_df):
    for column in train_df.columns:
        if train_df[column].dtype == "object" or not pd.api.types.is_numeric_dtype(
            train_df[column]
        ):
            train_df[column], test_df[column] = label_encode_column(
                train_df[column], test_df[column]
            )

    return train_df, test_df


def harmonize_series_to_categorical(
    series1: pd.Series, series2: pd.Series
) -> (pd.Series, pd.Series):
    """
    Convert two Pandas Series of type 'object' to categorical type with the same category values,
    which is the union of the values in both series.

    Parameters:
    series1 (pd.Series): The first input Series.
    series2 (pd.Series): The second input Series.

    Returns:
    tuple: A tuple containing the two Series converted to categorical type with the same categories.
    """
    # Get the union of unique values from both series
    combined_categories = pd.Series(pd.concat([series1, series2])).unique()

    # Convert the Series to categorical type with the combined categories
    series1_categorical = pd.Categorical(series1, categories=combined_categories)
    series2_categorical = pd.Categorical(series2, categories=combined_categories)

    return series1_categorical, series2_categorical


def prepare_xgb_data(
    dset: tableshift.core.tabular_dataset.Dataset,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_tr, y_tr, _, _ = dset.get_pandas("train")
    X_te, y_te, _, _ = dset.get_pandas("test")
    X_tr, X_te = tuple(bool_cols_to_categorical(df) for df in (X_tr, X_te))

    # "Target" is a special value in TableShift, and columns with this name will not be
    # cast to the correct type.
    for column in X_tr.columns:
        if "target" in column.lower() and X_tr[column].dtype == "object":
            X_tr[column], X_te[column] = harmonize_series_to_categorical(
                X_tr[column], X_te[column]
            )

    if is_categorical(y_tr):
        print("[DEBUG] casting y to numeric type.")
        y_tr, y_te = label_encode_column(y_tr, y_te)
    return X_tr, y_tr, X_te, y_te


def custom_encode(y_te: pd.Series, y_sample: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Re-encode y_te and y_sample such that the classes in y_sample are numbered consecutively from zero."""
    # Ensure inputs are numpy arrays
    y_te = np.array(y_te)
    y_sample = np.array(y_sample)

    values_to_map = list(set(x for x in y_te if x not in y_sample))

    # Find the maximum value across both arrays to start creating new values from
    max_value = max(np.max(y_te), np.max(y_sample))

    # Mapping for lower values to new higher class labels
    value_mapping = {
        old_value: i + max_value + 1 for i, old_value in enumerate(values_to_map)
    }

    # Apply mapping to y_te and y_sample
    adjusted_y_te = np.array([value_mapping.get(value, value) for value in y_te])
    adjusted_y_sample = np.array(
        [value_mapping.get(value, value) for value in y_sample]
    )

    # Encode the adjusted arrays
    le = LabelEncoder()
    le.fit(np.concatenate((adjusted_y_te, adjusted_y_sample)))
    y_te_encoded = le.transform(adjusted_y_te)
    y_sample_encoded = le.transform(adjusted_y_sample)
    return pd.Series(y_te_encoded), pd.Series(y_sample_encoded)
