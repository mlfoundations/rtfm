"""
Tests for the datasets module

To run tests: python -m unittest rtfm/tests/test_datasets.py -v

"""

import unittest

import numpy as np
import pandas as pd

from rtfm.datasets.data_utils import cast_columns_to_json_serializable, df_to_records


class TestCastJSONSerializable(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "date": [
                    np.datetime64("2005-02-25"),
                    np.datetime64("2005-02-25"),
                    np.datetime64("2005-02-25"),
                ],
                "delta": [
                    np.timedelta64(1, "D"),
                    np.timedelta64(1, "D"),
                    np.timedelta64(1, "D"),
                ],
                "timestamp": [
                    pd.Timestamp(1234),
                    pd.Timestamp(0),
                    pd.Timestamp(99),
                ],
                "datetime_utc": pd.to_datetime(
                    ["2024-01-01 12:00", "2024-01-02 12:00", "2024-01-02 12:00"],
                    utc=True,
                ),
                "timestamp_with_ints": [
                    pd.Timestamp(1234),
                    1234,
                    1234,
                ],
                "array_of_strings": [
                    ["a", "b", "c"],
                    ["x", np.nan, "y"],
                    ["a", "b", np.nan],
                ],
                "string_with_bytes": [
                    "my_string",
                    b"my_string",
                    "other_string",
                ],
                "categorical": pd.Categorical(["this", "that", "other"]),
                "string_with_nan": [np.nan, "my_string", "other_string"],
                "complex": [np.imag(0.1), np.imag(0.2), np.imag(0.3)],
            }
        )

    def test_cast_json_serializable(self):
        df = cast_columns_to_json_serializable(self.df)
        self.assertIsInstance(df, pd.DataFrame)
        records = df_to_records(df)
        self.assertTrue(all(isinstance(x, str) for x in records))

    def test_df_to_records(self):
        records = df_to_records(self.df)
        self.assertTrue(all(isinstance(x, str) for x in records))
