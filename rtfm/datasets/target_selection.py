from collections import defaultdict
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple, Union, Callable, Any, Literal, Callable

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from rtfm.configs import TargetSelectorConfig
from rtfm.datasets.data_utils import is_date_column, make_object_json_serializable
from tabliblib.summarizers import SingleColumnSummarizer

# Dummy logging statement to make sure linter doesn't remove logging module import.
logging.info("")


def is_numeric_series(vals: Union[pd.Series, Sequence[str]]) -> bool:
    return all(is_numeric(x) for x in vals)


@dataclass
class TargetSelector(ABC):
    config: TargetSelectorConfig

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Tuple[str, Sequence[str]]:
        """Select a target column from the columns in df.

        This method returns the name of the target column, and a Sequence of
        its unique values."""
        raise


@dataclass
class T4TargetSelector(TargetSelector):
    """Random selection from candidates meeting a set of heuristic criteria."""

    def __call__(
        self, df: pd.DataFrame, log_level: str = "warning"
    ) -> Tuple[str, Sequence[str]]:
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
                    self.config,
                    df[c],
                    unique_values_serializable,
                    log_level=log_level,
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

        p = self.config.labels_p_numeric
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
    config: TargetSelectorConfig,
    ser: pd.Series,
    unique_values_serializable: Sequence[str],
    log_level="warning",
) -> bool:
    """Check whether a target column is valid based on data_args."""
    log_fn = eval(f"logging.{log_level}")
    if "Unnamed:" in ser.name:
        log_fn(f"excluding target candidate {ser.name}")
        return False

    if config.labels_drop_dates and is_date_column(ser):
        log_fn(
            f"excluding target candidate {ser.name} due to being of date type {ser.dtype}."
        )
        return False

    if ser.nunique() < config.labels_min_unique_values:
        log_fn(
            f"excluding target candidate {ser.name} due to "
            f"insufficient number of unique values ({ser.nunique()} < data_args.labels_min_unique_values)"
        )
        return False

    all_values_are_numeric = is_numeric_series(unique_values_serializable)
    if (
        config.labels_require_nonunique
        and ser.nunique() == len(ser)
        # Allow numeric columns to have all unique values if labels_drop_numeric is False.
        and (not config.labels_drop_numeric and all_values_are_numeric)
    ):
        log_fn(f"excluding target candidate {ser.name} due to only unique values")
        return False

    if (
        config.labels_drop_numeric
        and all_values_are_numeric
        # Allow numeric columns if they are binary {0,1}.
        and not set(unique_values_serializable) == {"0", "1"}
    ):
        log_fn(f"excluding target candidate {ser.name} due to being of numeric type")
        return False

    if any(
        len(str(x)) > config.max_target_len_chars for x in unique_values_serializable
    ):
        log_fn(
            f"excluding target candidate {ser.name} due to values exceeding {config.max_target_len_chars} chars"
        )
        return False
    return True


def is_numeric(s) -> bool:
    """Check whether a string is numeric. This includes floats such as '3.5' and 3.'."""
    return bool(re.match(r"^-?\d+(\.+\d+)?$", s))


class NoTargetCandidatesError(ValueError):
    """Raised when there are no valid targets in a dataframe."""

    pass


@dataclass
class ModelBasedTargetSelector(TargetSelector):
    """Target selector that uses a model to select candidate columns."""

    clf = None
    summarizer: SingleColumnSummarizer = SingleColumnSummarizer(
        agg_fns={}, agg_quantiles=[], include_table_summary_metrics=False
    )

    def __post_init__(self):
        logging.warning(
            f"loading xgboost model from {self.config.model_path};"
            "if a segfault occurs try importing xgboost as early as possible"
            "in your code."
        )
        clf = XGBClassifier()
        clf.load_model(self.config.model_path)
        self.clf = clf

    def feature_extraction_fn(self, series: pd.Series) -> pd.Series:
        return self.summarizer(series)

    def __call__(
        self, df: pd.DataFrame, log_level: str = "warning"
    ) -> Tuple[str, Sequence[str]]:
        """Select a target column from the columns in df.

        This method returns the name of the target column, and a Sequence of
        its unique values."""

        column_scores = defaultdict(float)
        for colname in df.columns:
            features = self.feature_extraction_fn(df[colname])
            features = features[self.clf.get_booster().feature_names]
            score = self.clf.predict_proba([features]).flatten()
            if len(score) > 1:
                assert (
                    len(score) == 2
                ), f"expected scores of length 1 or 2, got {len(score)}"
                score = score[-1]
            column_scores[colname] = score
        # Sample score or return the highest, along with its unique serialized values
        scores_df = pd.DataFrame.from_dict(
            column_scores, orient="index", columns=["score"]
        )

        if self.config.selection_method == "max":
            selected_colname = scores_df.idxmax().item()

        elif self.config.selection_method == "topk":
            scores_topk = scores_df.score.nlargest(self.config.k)

            # Convert scores to np.float64 first to ensure probs
            # sum to exactly 1; otherwise it is possible to get
            # error in np.random.choice if sum is not close enough to 1.
            scores_topk_normalized = (
                np.array(scores_topk.values.astype(np.float64))
                / np.array(scores_topk.values.astype(np.float64)).sum()
            )

            selected_colname = np.random.choice(
                scores_topk.index, p=scores_topk_normalized
            )

        elif self.config.selection_method == "temperature":
            scores = scores_df.score.values.astype(np.float64)
            const = (scores ** (1 / self.config.temperature)).sum()
            scores = scores ** (1 / self.config.temperature) / const
            selected_colname = np.random.choice(scores_df.index, p=scores)

        else:
            raise NotImplementedError

        return selected_colname, df[selected_colname].unique().tolist()


def get_target_selector(config: TargetSelectorConfig):
    target_selector = eval(config.target_selector_cls)(config=config)
    return target_selector
