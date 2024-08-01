import json
from typing import Dict, Any, Tuple, Literal

import pandas as pd
import tableshift
import torch
from ray import tune
from sklearn.linear_model import LogisticRegression
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


#
# def get_categorical_idxs(df):
#     # Get a boolean mask for columns that are neither numeric nor boolean
#     categorical_mask = ~(df.dtypes.isin([np.number, bool]))
#
#     # Get the column indices where the mask is True
#     categorical_indices = np.where(categorical_mask)[0].tolist()
#
#     return categorical_indices
#
#
# def tune_catboost(X_tr, y_tr, num_trials: int):
#     def _optimize_hp(trial: optuna.trial.Trial, use_gpu=False, random_seed=42):
#         cb_params = {
#             # Same tuning grid as https://arxiv.org/abs/2106.11959,
#             # see supplementary section F.4.
#             "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
#             "depth": trial.suggest_int("depth", 3, 10),
#             "bagging_temperature": trial.suggest_float(
#                 "bagging_temperature", 1e-6, 1.0, log=True
#             ),
#             "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 100, log=True),
#             "leaf_estimation_iterations": trial.suggest_int(
#                 "leaf_estimation_iterations", 1, 10
#             ),
#             "use_best_model": True,
#             "task_type": "GPU" if use_gpu else "CPU",
#             "random_seed": random_seed,
#         }
#         model = CatBoostClassifier(**cb_params, cat_features=get_categorical_idxs(X_tr))
#         model.fit(X_tr, y_tr, verbose=False)
#         y_pred = model.predict(X_tr)
#         return accuracy_score(y_tr, y_pred)
#
#     study = optuna.create_study(direction="maximize")
#     study.optimize(_optimize_hp)
#     clf_with_best_params = CatBoostClassifier(**study.best_trial.params)
#     clf_with_best_params = clf_with_best_params.fit(X_tr, y_tr)
#     return clf_with_best_params


def get_categorical_idxs(df):
    # Get a boolean mask for columns that are neither numeric nor boolean
    categorical_mask = ~(df.dtypes.isin([np.number, bool]))

    # Get the column indices where the mask is True
    categorical_indices = np.where(categorical_mask)[0].tolist()

    return categorical_indices


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd


class ColumnCaster(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_cast):
        self.columns_to_cast = columns_to_cast
        self.target_dtype = str

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns_to_cast:
            X.iloc[:, col] = X.iloc[:, col].astype(self.target_dtype)
        return X


import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import Pipeline
import torch  # for GPU detection


def tune_catboost(
    X, y, n_iter=10, cv=3, catboost_iterations=64, catboost_early_stopping_rounds=5
):
    # Define parameter distributions
    param_distributions = {
        "learning_rate": np.logspace(-3, 0),
        "depth": np.arange(3, 11),
        "bagging_temperature": np.logspace(-6, 0),
        "l2_leaf_reg": np.logspace(0, 10),
        "leaf_estimation_iterations": np.arange(1, 10),
    }

    cat_features = get_categorical_idxs(X)

    # Catboost requires that all categorical features are of type string.
    column_caster = ColumnCaster(cat_features)
    X = column_caster.transform(X)

    cv = min(len(X), 3)

    # Check if GPU is available
    task_type = "GPU" if torch.cuda.is_available() and len(X) > cv else "CPU"
    print(f"CatBoost will use: {task_type}")

    n_classes = len(np.unique(y))
    if n_classes == 2:
        loss_function = "Logloss"
        eval_metric = "Logloss"
    elif n_classes > 2:
        loss_function = "MultiClass"
        eval_metric = "MultiClass"
    else:
        raise ValueError(f"got unexpected number of classes: {n_classes}")

    if len(X) > 1 and (all(y.value_counts() > cv)) and n_iter > 1:
        # Define the model
        model = CatBoostClassifier(
            iterations=catboost_iterations,
            early_stopping_rounds=catboost_early_stopping_rounds,
            random_state=42,
            verbose=1,
            task_type=task_type,  # Use GPU if available
            devices="0",  # Use first available GPU
            eval_metric=eval_metric,
            loss_function=loss_function,
        )

        # Create pools
        train_pool = Pool(X, y, cat_features=cat_features)

        # Perform randomized search
        grid_search_result = model.randomized_search(
            param_distributions,
            X=train_pool,
            y=None,
            n_iter=n_iter,
            cv=cv,
            refit=True,
            shuffle=True,
            verbose=False,
            plot=False,
        )
    else:
        model = CatBoostClassifier(
            iterations=catboost_iterations,
            early_stopping_rounds=catboost_early_stopping_rounds,
            random_state=42,
            verbose=1,
            cat_features=cat_features,
            task_type=task_type,  # Use GPU if available
            devices="0",  # Use first available GPU
            eval_metric=eval_metric,
            loss_function=loss_function,
        )
        model = model.fit(X, y)

    pipe = Pipeline([("caster", column_caster), ("catboost", model)])

    return pipe


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
    elif model_type == "catboost":
        clf = tune_catboost(X_tr, y_tr, n_iter=n_trials)

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
