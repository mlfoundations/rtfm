import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rtfm.arguments import DataArguments

from rtfm.datasets.data_utils import is_date_column, make_object_json_serializable


def is_numeric_series(vals: Union[pd.Series, Sequence[str]]) -> bool:
    return all(is_numeric(x) for x in vals)


@dataclass
class TargetSelector(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Tuple[str, Sequence[str]]:
        """Select a target column from the columns in df."""
        raise


@dataclass
class T4TargetSelector(TargetSelector):
    """Random selection from candidates meeting a set of heuristic criteria."""

    data_args: DataArguments

    def __call__(self, df: pd.DataFrame) -> Tuple[str, Sequence[str]]:
        # Iterate over the columns in the dataframe, and if it is a valid
        # candidate for being a target (more than one distinct value)
        # then add it to target_candidates_and_unique_values(). After this loop, target_candidates_and_unique_values
        # contains all target candidates.
        target_candidates_and_unique_values: Dict[str, pd.Series] = {}
        for c in df.columns:
            try:
                # Check that the values of the target column are not too long.
                unique_values_serializable = (
                    df[c].apply(make_object_json_serializable).unique()
                )

                if not is_valid_target_column(
                    self.data_args, df[c], unique_values_serializable
                ):
                    continue
                else:
                    target_candidates_and_unique_values[c] = unique_values_serializable

            except TypeError as te:
                # Case: there is an unhashable type in the targets, so it cannot
                # be counted with pd.Series.unique(); we do not consider these
                # as potential candidates to avoid typecasting the column.
                if "unhashable type" in str(te):
                    continue
                else:
                    raise te

        # Compute weighted probabilities for the target candidates.
        target_candidates = list(target_candidates_and_unique_values.keys())
        numeric_count = sum(
            is_numeric_series(vals)
            for vals in target_candidates_and_unique_values.values()
        )
        nonnumeric_count = len(target_candidates) - numeric_count

        p = self.data_args.labels_p_numeric
        target_probs = (
            [
                p / numeric_count
                if is_numeric_series(vals)
                else (1 - p) / nonnumeric_count
                for vals in target_candidates_and_unique_values.values()
            ]
            if numeric_count and nonnumeric_count
            else None
        )
        if target_probs:
            target_probs = np.array(target_probs) / sum(target_probs)

        if not target_candidates:
            raise NoTargetCandidatesError
        # Choose a target uniformly at random for the table. This target will be used for all examples in the table.
        target = np.random.choice(target_candidates, p=target_probs)
        return target, target_candidates_and_unique_values[target]


def is_valid_target_column(
    data_args: DataArguments, ser: pd.Series, unique_values_serializable: Sequence[str]
) -> bool:
    """Check whether a target column is valid based on data_args."""
    if "Unnamed:" in ser.name:
        logging.warning(f"excluding target candidate {ser.name}")
        return False

    if data_args.labels_drop_dates and is_date_column(ser):
        logging.warning(
            f"excluding target candidate {ser.name} due to being of date type {ser.dtype}."
        )
        return False

    if ser.nunique() < data_args.labels_min_unique_values:
        logging.warning(
            f"excluding target candidate {ser.name} due to "
            f"insufficient number of unique values ({ser.nunique()} < data_args.labels_min_unique_values)"
        )
        return False

    all_values_are_numeric = is_numeric_series(unique_values_serializable)
    if (
        data_args.labels_require_nonunique
        and ser.nunique() == len(ser)
        # Allow numeric columns to have all unique values if labels_drop_numeric is False.
        and (not data_args.labels_drop_numeric and all_values_are_numeric)
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to only unique values"
        )
        return False

    if (
        data_args.labels_drop_numeric
        and all_values_are_numeric
        # Allow numeric columns if they are binary {0,1}.
        and not set(unique_values_serializable) == {"0", "1"}
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to being of numeric type"
        )
        return False

    if any(
        len(str(x)) > data_args.max_target_len_chars for x in unique_values_serializable
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to values exceeding {data_args.max_target_len_chars} chars"
        )
        return False
    return True


def is_numeric(s) -> bool:
    """Check whether a string is numeric. This includes floats such as '3.5' and 3.'."""
    return bool(re.match(r"^-?\d+(\.+\d+)?$", s))


class NoTargetCandidatesError(ValueError):
    """Raised when there are no valid targets in a dataframe."""

    pass
