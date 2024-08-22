"""
Test for target selection.

To run tests:
python -m unittest rtfm/tests/test_target_selector.py -v

"""
import unittest
from datetime import datetime

from xgboost import XGBClassifier

import numpy as np
import pandas as pd
from tabliblib.summarizers import SingleColumnSummarizer

from rtfm.configs import TargetConfig
from rtfm.datasets.target_selection import T4TargetSelector, ModelBasedTargetSelector

# Set the random seed as some tests rely on random selection
np.random.seed(42)


class TestT4TargetSelector(unittest.TestCase):
    def setUp(self):
        self.config = TargetConfig(target_selector_cls="T4TargetSelector")
        self.target_selector = T4TargetSelector(self.config)

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
            config=TargetConfig(
                target_selector_cls="T4TargetSelector", labels_require_nonunique=False
            )
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
            config=TargetConfig(
                target_selector_cls="T4TargetSelector", labels_drop_dates=False
            )
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
    def test_model_based_selector(
        self,
    ):
        """Simple integration test of model-based selector."""
        config = TargetConfig(target_selector_cls="ModelBasedTargetSelector")
        selector = ModelBasedTargetSelector(config=config)
        assert isinstance(selector.clf, XGBClassifier)

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
        config = TargetConfig(
            target_selector_cls="ModelBasedTargetSelector",
            selection_method="topk",
            k=2,
        )
        selector = ModelBasedTargetSelector(config=config)

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
        config = TargetConfig(
            target_selector_cls="ModelBasedTargetSelector",
            selection_method="temperature",
        )

        selector = ModelBasedTargetSelector(config=config)

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

        # Since we cannot guarantee all columns will always be selected (this depends
        # on the classifier being used), we at least check that more than one column
        # is selected over the 1000 trials.
        self.assertGreaterEqual(len(selected_columns), 2)
