"""
Test for target selection.

To run tests:
python -m unittest rtfm/tests/test_target_selector.py -v

"""
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from tabliblib.summarizers import SingleColumnSummarizer
from xgboost import XGBClassifier

from rtfm.arguments import DataArguments
from rtfm.datasets.target_selection import T4TargetSelector, ModelBasedTargetSelector

# Set the random seed as some tests rely on random selection
np.random.seed(42)


class TestT4TargetSelector(unittest.TestCase):
    def setUp(self):
        self.data_args = DataArguments()
        self.target_selector = T4TargetSelector(self.data_args)

    def test_t4_settings(self, num_trials=100):
        """Test that T4 inclusion/exclusion rules are applied on dummy data."""

        df = pd.DataFrame(
            {
                # Requires nonunique values, so we sample to ensure repeats
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.random.choice(np.arange(4), 10).astype(int),
                "my_float": np.random.choice(np.arange(4), 10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        selected = set()

        for _ in range(num_trials):
            target_col, _ = self.target_selector(df, log_level="info")
            selected.add(target_col)
        self.assertSetEqual(selected, {"my_str", "my_int", "my_float"})

    def test_exclude_unique_numeric_columns(self, num_trials=100):
        """Test that unique numeric columns are dropped."""
        df = pd.DataFrame(
            {
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.arange(10).astype(int),
                "my_float": np.arange(10).astype(float),
            }
        )
        selected = set()

        for _ in range(num_trials):
            target_col, _ = self.target_selector(df, log_level="info")
            selected.add(target_col)
        self.assertSetEqual(
            selected,
            {
                "my_str",
            },
        )

    def test_allow_unique_columns(self, num_trials=100):
        """Test that labels_require_nonunique=False allows unique-valued columns."""
        _target_selector = T4TargetSelector(
            data_args=DataArguments(labels_require_nonunique=False)
        )
        df = pd.DataFrame(
            {
                "my_str": list("abcdefghij"),
                "my_int": np.arange(10).astype(int),
                "my_float": np.arange(10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        selected = set()

        for _ in range(num_trials):
            target_col, _ = _target_selector(df, log_level="info")
            selected.add(target_col)
        self.assertSetEqual(selected, {"my_str", "my_int", "my_float"})

    def test_allow_datetime_columns(self, num_trials=100):
        """Test that labels_drop_dates=False allows DateTime columns."""
        _target_selector = T4TargetSelector(
            data_args=DataArguments(labels_drop_dates=False)
        )
        df = pd.DataFrame(
            {
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.random.choice(np.arange(4), 10).astype(int),
                "my_float": np.random.choice(np.arange(4), 10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        selected = set()

        for _ in range(num_trials):
            target_col, _ = _target_selector(df, log_level="info")
            selected.add(target_col)
        self.assertSetEqual(selected, {"my_str", "my_int", "my_float", "my_datetime"})


class TestModelBasedTargetSelector(unittest.TestCase):
    def setUp(self):
        self.data_args = DataArguments()
        model_path = "/Users/jpgard/Documents/github/tabliblib-official/tabliblib/xgb_target_scorer.json"

        clf = XGBClassifier()
        clf.load_model(model_path)
        self.clf = clf
        self.summarizer = SingleColumnSummarizer(
            agg_fns={}, agg_quantiles=[], include_table_summary_metrics=False
        )

    def test_model_based_selector(
        self,
    ):
        """Simple integration test of model-based selector."""

        selector = ModelBasedTargetSelector(
            data_args=self.data_args,
            feature_extraction_fn=lambda series: self.summarizer(pd.DataFrame(series)),
            classifier=self.clf,
        )

        df = pd.DataFrame(
            {
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.random.choice(np.arange(4), 10).astype(int),
                "my_float": np.random.choice(np.arange(4), 10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        target, _ = selector(df, log_level="info")

    def test_model_based_selector_topk(
        self,
    ):
        """Check that model-based selector with top-k chooses all k columns."""

        selector = ModelBasedTargetSelector(
            data_args=self.data_args,
            feature_extraction_fn=lambda series: self.summarizer(pd.DataFrame(series)),
            classifier=self.clf,
            selection_method="topk",
            k=2,
        )

        df = pd.DataFrame(
            {
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.random.choice(np.arange(4), 10).astype(int),
                "my_float": np.random.choice(np.arange(4), 10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        selected_columns = set()
        for _ in range(1000):
            target, _ = selector(df, log_level="info")
            selected_columns.add(target)

        self.assertEqual(len(selected_columns), 2)

    def test_model_based_selector_temperature(
        self,
    ):
        """Check that model-based selector with temperature chooses all columns."""

        selector = ModelBasedTargetSelector(
            data_args=self.data_args,
            feature_extraction_fn=lambda series: self.summarizer(pd.DataFrame(series)),
            classifier=self.clf,
            selection_method="temperature",
        )

        df = pd.DataFrame(
            {
                "my_str": np.random.choice(list("abcdefg"), 10),
                "my_int": np.random.choice(np.arange(4), 10).astype(int),
                "my_float": np.random.choice(np.arange(4), 10).astype(float),
                "my_datetime": [datetime.now() for _ in range(10)],
            }
        )
        selected_columns = set()
        for _ in range(1000):
            target, _ = selector(df, log_level="info")
            selected_columns.add(target)

        self.assertEqual(len(selected_columns), 4)
