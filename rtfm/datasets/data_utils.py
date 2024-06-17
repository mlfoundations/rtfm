import datetime
import json
import logging
from typing import Union, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tableshift.core.features import get_numeric_columns

from rtfm.arguments import DataArguments


def transform_numeric_columns(df, transform: str) -> pd.DataFrame:
    # Only the unmapped numeric columns will keep a numeric data type; all others
    # are cast to dtype 'object' by sklearn when the mapping is applied.
    numeric_cols = get_numeric_columns(df)
    if transform == "quantile":
        transforms = [
            (f"quantile_{c}", QuantileTransformer(), [c]) for c in numeric_cols
        ]
    elif transform == "scale":
        transforms = [(f"scale_{c}", StandardScaler(), [c]) for c in numeric_cols]
    else:
        raise ValueError(f"unknown transform: {transform}")
    feature_transformer = ColumnTransformer(
        transforms,
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    transformed = feature_transformer.fit_transform(df)
    transformed = pd.DataFrame(
        transformed, columns=feature_transformer.get_feature_names_out()
    )
    return transformed


class CannotDecodeBytesError(ValueError):
    pass


def bytes_to_str(byte_data):
    encodings = ["utf-8", "ascii", "iso-8859-1"]
    for encoding in encodings:
        try:
            return byte_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    # If all decodings fail, raise an exception or return a default value
    raise CannotDecodeBytesError("Failed to decode bytes using common encodings.")


def is_array_type(x) -> bool:
    return (
            isinstance(x, np.ndarray)
            or isinstance(x, pd.Series)
            or isinstance(x, list)
            or isinstance(x, tuple)
    )


def make_object_json_serializable(x) -> Union[str, List[str]]:
    """Make x JSON serializable. If x is an array-like, make all of its elements JSON serialiable."""
    if is_array_type(x):
        return [make_object_json_serializable(elem) for elem in x]
    if pd.isnull(x):
        return str(x)
    elif isinstance(x, str):
        return x
    elif isinstance(x, bytes):
        return bytes_to_str(x)
    elif isinstance(
            x, (datetime.time, pd.Timestamp, pd.Timedelta, pd.Period, np.datetime64)
    ):
        # Explicitly convert datetime objects to string, ensuring timezone-aware datetimes are handled.
        return x.isoformat() if hasattr(x, "isoformat") else str(x)
    else:
        return str(x)


def is_date_column(ser: pd.Series) -> bool:
    """More robust check of whether a column contains a date.

    Specifically, pd.core.dtypes.common.is_datetime_or_timedelta_dtype does not handle
    datetime64tz types (it returns False) so we explicitly include that check."""
    return (
            pd.core.dtypes.common.is_datetime_or_timedelta_dtype(ser)
            or pd.core.dtypes.common.is_datetime64_dtype(ser)
            or pd.core.dtypes.common.is_datetime64tz_dtype(ser)
    )


def cast_columns_to_json_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """Cast any columns of data types that are not JSON-serializable to a JSON-serializable type."""
    columns_to_drop = []
    for c in df.columns:
        # Handle timestamp (np.datetime64, np.timedelta64) types
        if is_date_column(df[c]):
            logging.warning(
                f"casting column of type {df[c].dtype} to JSON-serializable str"
            )
            df[c] = df[c].apply(make_object_json_serializable)

        # Handle bytes values
        elif df[c].dtype == "object":
            # Check if there are any bytes values in the column
            if not all(isinstance(x, str) for x in df[c]):
                logging.warning(
                    f"casting values of column {c} to JSON-serializable str"
                )
                # Convert bytes to strings, leave other types as is
                try:
                    df[c] = df[c].apply(make_object_json_serializable)
                except CannotDecodeBytesError:
                    # Drop the column if we cannot decode the bytes.
                    logging.warning(
                        f"Could not decode bytes in column {c}; dropping it."
                    )
                    columns_to_drop.append(c)

        # Check for categorical data
        elif df[c].dtype.name == "category":
            df[c] = df[c].astype(str)

        # Check for complex data types
        elif df[c].dtype == "complex":
            df[c] = df[c].apply(make_object_json_serializable)
    df.drop(columns=columns_to_drop, inplace=True)
    return df


def df_to_records(df) -> pd.DataFrame:
    df = cast_columns_to_json_serializable(df)
    records = df.to_dict(orient="records")
    df_out = DataFrame({"data": [json.dumps(x) for x in records]})
    return df_out


def build_formatted_df(df, ds_info: str, data_args: DataArguments) -> DataFrame:
    """Get a DataFrame that is reformatted for use with the training loop.

    Specifically, this produces as dataset with at least two columns: 'info' and 'data'.
    Each row in the resulting DataFrame represents one observation.
    The 'info' column contains the dataset metadata,
    while 'data' contains the feature names
    and values associated with the observation.
    If the training run uses in-task_context learning, in-task_context examples are added
    as specified in the data_arguments in a third column, 'in_context_examples'.
    """

    if data_args.shuffle_table_features:
        df = df.sample(frac=1, axis=1)

    df_out = df_to_records(df)
    df_out["info"] = ds_info

    meta = {}
    if data_args.use_metafeatures:
        for transform in ("quantile", "scale"):
            transformed = transform_numeric_columns(df, transform)
            transformed = transformed.round(data_args.metafeatures_max_precision)
            assert all(x in df.columns for x in transformed.columns)
            meta[transform] = transformed.to_dict(orient="records")
        # meta is a dict of lists; convert it to a list of dicts
        meta = [dict(zip(meta.keys(), elem)) for elem in zip(*meta.values())]

        # The '__metafeatures__' field is a nested dictionary. For each key ('quantile', 'scale', etc.)
        # the value is a dictionary; the keys of this nested dictionary are column names
        # which will also appear in the 'data' field, and the values of this nested dictionary
        # are the corresponding values of the metafeature (e.g. the quantile or scaled value).
        if meta:
            # Even if use_metafeatures is true, for some datasets there are no quantitative features;
            # this block skipps adding metafeatures in that casel
            df_out["__metafeatures__"] = meta
    return df_out
