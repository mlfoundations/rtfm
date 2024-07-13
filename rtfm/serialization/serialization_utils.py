import re
from typing import Union, Dict, Any

import numpy as np
import pandas as pd


def shuffle_example_features(
    x: Union[pd.Series, Dict[Any, Any]]
) -> Union[pd.Series, Dict[Any, Any]]:
    """Shuffle the order of features in a sample."""
    # Randomly the order of features in x.
    keys = list(x.keys())
    np.random.shuffle(keys)
    return {k: x[k] for k in keys}


def apply_feature_dropout(
    x: Union[pd.Series, Dict[Any, Any]], p: float
) -> Union[pd.Series, Dict[Any, Any]]:
    """Randomly drop out key-value pairs from a sample with probability p."""
    return {k: v for k, v in x.items() if p < np.random.uniform()}


def find_all_idxs(substring, target_string):
    """Find the starting index of every instance of substring in target_string."""
    escaped_substring = re.escape(substring)
    return sorted([m.start() for m in re.finditer(escaped_substring, target_string)])


def extract_metafeatures(
    feature_name: str, metafeatures: Dict[str, Dict[str, Any]]
) -> Union[Dict[str, Any], None]:
    """Parse the metafeatures and fetch a metafeatures dict for the feature,
    which maps metafeature names to values.

    Example output: {'quantile': 0.95, 'scale': 1.96}

    Note that all metafeatures returned by this fucntion will always have non-null values; any
    metafeatures with a value of None will not be included in the output.
    """
    if (metafeatures is not None) and any(
        feature_name in meta_feature_dict for meta_feature_dict in metafeatures.values()
    ):
        return {
            k: v[feature_name] for k, v in metafeatures.items() if feature_name in v
        }


def strip_html_whitespace(x: str) -> str:
    return re.sub(">\s+<", "><", x)


def discretize_continuous_column(column: pd.Series, num_buckets: int = 4) -> pd.Series:
    """Take a continuous-valued column and discretize it into num_buckets.

    The formatting of the outputs is of the form 'less than 0', 'between 0.5 and 0.9', or 'greater than 1.99'
    etc. depending on the value of the observation and the number of buckets used.

    This is the same format used in training TabuLa-8B and should be used for inference and
    evaluation of that model.
    """
    assert pd.api.types.is_numeric_dtype(column)
    assert num_buckets > 0, "Number of buckets must be greater than zero."

    # Compute bucket thresholds
    thresholds = [column.quantile(i / num_buckets) for i in range(1, num_buckets)]

    # Define a function to categorize each value
    def categorize_value(x):
        for i, threshold in enumerate(thresholds):
            if x < threshold:
                if i == 0:
                    return f"less than {threshold}"
                else:
                    return f"between {thresholds[i-1]} and {threshold}"
        return f"greater than {thresholds[-1]}"

    # Apply the categorization function to the column
    return column.apply(categorize_value)
