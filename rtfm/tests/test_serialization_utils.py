"""
Tests for serialization utils.

To run tests: python -m unittest rtfm/tests/test_serialization_utils.py -v
"""
import unittest

from rtfm.serialization.serialization_utils import (
    shuffle_example_features,
    apply_feature_dropout,
    extract_metafeatures,
)


class TestShuffleFeatures(unittest.TestCase):
    def test_shuffle_features(self, num_tries=100):
        example = {"a": 1, "b": 2.0, "c": "3"}
        shuffled = [shuffle_example_features(example) for _ in range(num_tries)]
        self.assertFalse(
            all(
                list(shuffled[0].items()) == list(shuffled[i].items())
                for i in range(num_tries)
            )
        )


class TestFeatureDropout(unittest.TestCase):
    def test_feature_dropout(self, num_tries=100_000, p=0.5):
        example = {"a": 1, "b": 2.0, "c": "3"}
        dropped = [apply_feature_dropout(example, p) for _ in range(num_tries)]
        p_empirical = sum(len(x) for x in dropped) / (num_tries * len(example))
        self.assertAlmostEqual(p_empirical, p, places=2)


class TestExtractMetafeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = {
            "quantile": {"x1": 0.92, "x2": 0.25},
            "scale": {"x1": 1.9, "x2": -0.35},
        }

    def test_extract_metafeatures_valid(self):
        """Test extraction with a simple example"""
        result = extract_metafeatures("x1", self.meta)
        expected = {"quantile": 0.92, "scale": 1.9}
        self.assertDictEqual(result, expected)

        result = extract_metafeatures("x2", self.meta)
        expected = {"quantile": 0.25, "scale": -0.35}
        self.assertDictEqual(result, expected)

    def test_extract_metafeatures_missing(self):
        """Test extraction with a feature with no metafeatures"""
        result = extract_metafeatures("x3", self.meta)
        self.assertIsNone(result)
