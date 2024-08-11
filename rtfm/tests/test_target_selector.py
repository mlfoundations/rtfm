"""
Test for target selection.

To run tests:
python -m unittest rtfm/tests/test_target_selector.py -v

"""
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from rtfm.arguments import DataArguments
from rtfm.datasets.target_selection import T4TargetSelector


class TestT4TargetSelector(unittest.TestCase):
    def setUp(self):
        self.data_args = DataArguments()
        self.target_selector = T4TargetSelector(self.data_args, log_level="info")

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
            target_col, _ = self.target_selector(df)
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
            target_col, _ = self.target_selector(df)
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
            data_args=DataArguments(labels_require_nonunique=False), log_level="info"
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
            target_col, _ = _target_selector(df)
            selected.add(target_col)
        self.assertSetEqual(selected, {"my_str", "my_int", "my_float"})

    def test_allow_datetime_columns(self, num_trials=100):
        """Test that labels_drop_dates=False allows DateTime columns."""
        _target_selector = T4TargetSelector(
            data_args=DataArguments(labels_drop_dates=False), log_level="info"
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
            target_col, _ = _target_selector(df)
            selected.add(target_col)
        self.assertSetEqual(selected, {"my_str", "my_int", "my_float", "my_datetime"})
