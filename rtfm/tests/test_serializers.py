"""
Tests for serialization.

To run tests: python -m unittest rtfm/tests/test_serializers.py -v
"""
import json
import re
import unittest

import pandas as pd

from rtfm.configs import SerializerConfig
from rtfm.serialization.serializers import (
    BasicSerializer,
    BasicSerializerV2,
    StructuredSerializer,
    PandasSeriesSerializer,
    HtmlSerializer,
    JsonSerializer,
    HtmlNoWhitespaceSerializer,
)


class TestBasicSerializer(unittest.TestCase):
    def test_basic_serializer_no_prefix_suffix_choices(self):
        """Test BasicSerializer with only feature keys/values."""
        serializer = BasicSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        expected = "The float_feature is -768.25. The int_feature is -25. The str_feature is my_category."
        serialized = serializer(dummy_input)
        self.assertEqual(serialized, expected)

    def test_basic_serializer_with_prefix_suffix_choices(self):
        """Test BasicSerializer with prefix, suffix, and choices."""
        serializer = BasicSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0"
        dummy_serialized_expected = "The float_feature is -768.25. The int_feature is 5968. The str_feature is my_category."

        # Expected: 'This is the task context. This is an observation. 1 or 0.
        #   The float_feature is -768.25. The int_feature is 5968.
        #   The str_feature is my_category. What is the label?'
        expected = " ".join(
            x.strip()
            for x in [
                task_context_text,
                prefix_text,
                choices_text,
                dummy_serialized_expected,
                suffix_text,
                choices_text,
            ]
        ).strip()

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["1", "0"],
            task_context_text=task_context_text,
        )
        self.assertEqual(serialized, expected)

    def test_basic_serializer_raises_on_bad_suffix(self):
        """Test that BasicSerializer raises ValueError when the suffix contains a period."""
        serializer = BasicSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -7168.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "The suffix has a period."
        choices_text = "1 or 0."
        with self.assertRaises(ValueError):
            serializer(
                dummy_input,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                choices_text=choices_text,
                task_context_text=task_context_text,
                strict=True,
            )

    def test_basic_serializer_raises_on_bad_keys(self):
        """Test that BasicSerializer raises ValueError when one key is prefix of another."""
        serializer = BasicSerializer(config=SerializerConfig(), strict=True)
        dummy_input = {
            "float_feature": -7168.25,
            "float_feature_2": -68.99,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0."
        with self.assertRaises(ValueError):
            serializer(
                dummy_input,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                choices_text=choices_text,
                task_context_text=task_context_text,
            )

    def test_basic_serialize_deserialize(self):
        serializer = BasicSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -7168.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0."
        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices_text=choices_text,
            task_context_text=task_context_text,
        )
        feature_names = list(dummy_input.keys())
        deserialized = serializer.deserialize_example(serialized, feature_names)
        deserialized_expected = {k: str(v) for k, v in dummy_input.items()}
        self.assertDictEqual(deserialized, deserialized_expected)

    def test_basic_serializer_with_shuffle(self, num_tries=100):
        """Test that shuffling occurs properly."""
        serializer = BasicSerializer(
            config=SerializerConfig(shuffle_instance_features=True)
        )
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        serialized = [serializer(dummy_input) for _ in range(num_tries)]

        # Check that the serialized examples are not all identical.
        self.assertFalse(all(serialized[i] == serialized[0] for i in range(num_tries)))

    def test_basic_serializer_with_dropout(self, num_tries=100_000, p=0.5):
        """Test that dropout occurs properly."""
        serializer = BasicSerializer(config=SerializerConfig(feature_dropout=p))
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "zero_feature": 0,
            "str_feature": "my_category",
            "bool_feature": True,
        }
        serialized = [serializer(dummy_input) for _ in range(num_tries)]

        # Check that the overall rate of features being present is around p.
        feature_in_x_rate = sum(
            [sum(feat not in ser for feat in dummy_input.keys()) for ser in serialized]
        ) / (len(dummy_input) * num_tries)
        self.assertAlmostEqual(feature_in_x_rate, p, 2)

    def test_basic_serializer_with_meta(self):
        """Test the BasicSerializer with metafeatures."""
        serializer = BasicSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "zero_feature": 0,
            "str_feature": "my_category",
            "bool_feature": True,
        }
        meta = {
            "quantile": {
                "float_feature": 0.29,
                "int_feature": 0.99,
                "zero_feature": 0.1,
            },
            "scale": {
                "float_feature": -0.2,
                "int_feature": 1.24,
                "zero_feature": 0.0,
            },
        }
        serialized = serializer(dummy_input, meta=meta)
        expected = "The float_feature is -768.25 (quantile:0.29, scale:-0.2). The int_feature is -25 (quantile:0.99, scale:1.24). The zero_feature is 0 (quantile:0.1, scale:0.0). The str_feature is my_category. The bool_feature is True."
        self.assertEqual(serialized, expected)

    def test_basic_serializer_with_dropout_and_meta(self, num_tries=100_000, p=0.5):
        """Test that dropout occurs properly with metafeatures."""
        serializer = BasicSerializer(config=SerializerConfig(feature_dropout=p))
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "zero_feature": 0,
            "str_feature": "my_category",
            "bool_feature": True,
        }
        meta = {
            "quantile": {
                "float_feature": 0.09,
                "int_feature": 0.91,
                "zero_feature": 0.01,
            },
            "scale": {
                "float_feature": -0.52,
                "int_feature": -1.4,
                "zero_feature": 0.0,
            },
        }
        serialized = [serializer(dummy_input, meta=meta) for _ in range(num_tries)]

        # Check that the overall rate of features being present is around p.
        feature_in_x_rate = sum(
            [sum(feat not in ser for feat in dummy_input.keys()) for ser in serialized]
        ) / (len(dummy_input) * num_tries)
        self.assertAlmostEqual(feature_in_x_rate, p, 2)

    def test_basic_serializer_special_tokens(self):
        """Test that BasicSerializer special tokens map is not empty."""
        serializer = BasicSerializer(config=SerializerConfig())
        self.assertTrue(len(serializer.special_tokens) > 0)


class TestBasicSerializerV2(unittest.TestCase):
    def test_with_prefix_suffix_choices(self):
        """Test BasicSerializerV2 with prefix, suffix, and choices."""
        serializer = BasicSerializerV2(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "||1||0||"
        dummy_serialized_expected = "The float_feature is -768.25. The int_feature is 5968. The str_feature is my_category."

        # Expected: 'This is the task context. This is an observation. 1 or 0.
        #   The float_feature is -768.25. The int_feature is 5968.
        #   The str_feature is my_category. What is the label?'
        expected = " ".join(
            x.strip()
            for x in [
                task_context_text,
                prefix_text,
                choices_text,
                dummy_serialized_expected,
                suffix_text,
                choices_text,
            ]
        ).strip()

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["1", "0"],
            task_context_text=task_context_text,
        )

        self.assertEqual(serialized, expected)

    def test_with_choices_front(self):
        """Test BasicSerializerV2 with prefix, suffix, and choices."""
        serializer = BasicSerializerV2(
            config=SerializerConfig(choices_position="front")
        )
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "||1||0||"
        dummy_serialized_expected = "The float_feature is -768.25. The int_feature is 5968. The str_feature is my_category."

        # Expected: 'This is the task context. This is an observation. 1 or 0.
        #   The float_feature is -768.25. The int_feature is 5968.
        #   The str_feature is my_category. What is the label?'
        expected = " ".join(
            x.strip()
            for x in [
                task_context_text,
                prefix_text,
                choices_text,
                dummy_serialized_expected,
                suffix_text,
            ]
        ).strip()

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["1", "0"],
            task_context_text=task_context_text,
        )

        self.assertEqual(serialized, expected)

    def test_with_choices_back(self):
        """Test BasicSerializerV2 with prefix, suffix, and choices."""
        serializer = BasicSerializerV2(config=SerializerConfig(choices_position="back"))
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "||1||0||"
        dummy_serialized_expected = "The float_feature is -768.25. The int_feature is 5968. The str_feature is my_category."

        # Expected: 'This is the task context. This is an observation. 1 or 0.
        #   The float_feature is -768.25. The int_feature is 5968.
        #   The str_feature is my_category. What is the label?'
        expected = " ".join(
            x.strip()
            for x in [
                task_context_text,
                prefix_text,
                dummy_serialized_expected,
                suffix_text,
                choices_text,
            ]
        ).strip()

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["1", "0"],
            task_context_text=task_context_text,
        )

        self.assertEqual(serialized, expected)


class TestStructuredSerializer(unittest.TestCase):
    def test_structured_serializer(self):
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = serializer(dummy_input)
        expected = (
            "<|EXAMPLE|><|KEY|>float_feature<|/KEY|><|VALUE|>-99.5<|/VALUE|>"
            "<|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|>"
            "<|KEY|>str_feature<|/KEY|><|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_choices_both(self):
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = serializer(dummy_input, choices=["0", "1"])
        expected = (
            "<|CHOICES|>0||1<|/CHOICES|><|EXAMPLE|><|KEY|>float_feature<|/KEY|>"
            "<|VALUE|>-99.5<|/VALUE|><|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|>"
            "<|KEY|>str_feature<|/KEY|><|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
            "<|CHOICES|>0||1<|/CHOICES|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_choices_front(self):
        serializer = StructuredSerializer(
            config=SerializerConfig(choices_position="front")
        )
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = serializer(dummy_input, choices=["0", "1"])
        expected = (
            "<|CHOICES|>0||1<|/CHOICES|><|EXAMPLE|><|KEY|>float_feature<|/KEY|>"
            "<|VALUE|>-99.5<|/VALUE|><|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|>"
            "<|KEY|>str_feature<|/KEY|><|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_choices_back(self):
        serializer = StructuredSerializer(
            config=SerializerConfig(choices_position="back")
        )
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = serializer(dummy_input, choices=["0", "1"])
        expected = (
            "<|EXAMPLE|><|KEY|>float_feature<|/KEY|>"
            "<|VALUE|>-99.5<|/VALUE|><|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|>"
            "<|KEY|>str_feature<|/KEY|><|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
            "<|CHOICES|>0||1<|/CHOICES|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_prefix_suffix_choices(self):
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = serializer(
            dummy_input,
            choices=["0", "1"],
            prefix_text="Predict the target:",
            suffix_text="What is the value of target?",
        )
        expected = (
            "<|PREFIX|>Predict the target:<|/PREFIX|><|CHOICES|>0||1<|/CHOICES|>"
            "<|EXAMPLE|><|KEY|>float_feature<|/KEY|><|VALUE|>-99.5<|/VALUE|>"
            "<|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|><|KEY|>str_feature<|/KEY|>"
            "<|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
            "<|SUFFIX|>What is the value of target?<|/SUFFIX|>"
            "<|CHOICES|>0||1<|/CHOICES|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_meta(self):
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        meta = {
            "quantile": {"float_feature": 0.99, "int_feature": 0.91},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }
        serialized = serializer(dummy_input, meta=meta)
        expected = (
            "<|EXAMPLE|><|KEY|>float_feature<|/KEY|><|VALUE|>-99.5<|/VALUE|>"
            "<|META|><|QUANTILE|>0.99<|/QUANTILE|><|SCALE|>-0.2<|/SCALE|><|/META|>"
            "<|KEY|>int_feature<|/KEY|><|VALUE|>-1<|/VALUE|><|META|>"
            "<|QUANTILE|>0.91<|/QUANTILE|><|SCALE|>-0.99<|/SCALE|><|/META|>"
            "<|KEY|>str_feature<|/KEY|><|VALUE|>my_category<|/VALUE|><|/EXAMPLE|>"
        )
        self.assertEqual(serialized, expected)

    def test_structured_serializer_with_shuffle(self, num_tries=100):
        serializer = StructuredSerializer(
            config=SerializerConfig(shuffle_instance_features=True)
        )
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        serialized = [serializer(dummy_input) for _ in range(num_tries)]
        # Check that the serialized examples are not all identical.
        self.assertFalse(all(serialized[i] == serialized[0] for i in range(num_tries)))

    def test_structured_serializer_with_dropout(self, num_tries=100_000, p=0.5):
        """Test that dropout occurs properly."""
        serializer = StructuredSerializer(config=SerializerConfig(feature_dropout=p))
        dummy_input = {
            "float_feature": -18.5,
            "int_feature": 105,
            "zero_feature": 0,
            "str_feature": "my_category",
            "bool_feature": False,
        }
        serialized = [serializer(dummy_input) for _ in range(num_tries)]

        # Check that the overall rate of features being present is around p.
        feature_in_x_rate = sum(
            [sum(feat not in ser for feat in dummy_input.keys()) for ser in serialized]
        ) / (len(dummy_input) * num_tries)
        self.assertAlmostEqual(feature_in_x_rate, p, 2)

    def test_structured_serializer_with_dropout_and_meta(
        self, num_tries=100_000, p=0.5
    ):
        """Test that dropout occurs properly."""
        serializer = StructuredSerializer(config=SerializerConfig(feature_dropout=p))
        dummy_input = {
            "float_feature": -18.5,
            "int_feature": 105,
            "zero_feature": 0,
            "str_feature": "my_category",
            "bool_feature": False,
        }
        meta = {
            "quantile": {"float_feature": 0.99, "int_feature": 0.91},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }

        serialized = [serializer(dummy_input, meta=meta) for _ in range(num_tries)]

        # Check that the overall rate of features being present is around p.
        feature_in_x_rate = sum(
            [sum(feat not in ser for feat in dummy_input.keys()) for ser in serialized]
        ) / (len(dummy_input) * num_tries)
        self.assertAlmostEqual(feature_in_x_rate, p, 2)

    def test_structured_serializer_special_tokens(self):
        """Test that StructuredSerializer special tokens map is not empty."""
        serializer = StructuredSerializer(config=SerializerConfig())
        self.assertTrue(len(serializer.special_tokens) > 0)

    def test_structured_serializer_raises_on_bad_keys(self):
        """Test that StructuredSerializer raises ValueError when one key is prefix of another.

        (This is probably ok for StructuredSerializer but we enforce it to be safe, and because
        this shouldn't occur under normal circumstances.)"""
        serializer = StructuredSerializer(config=SerializerConfig(), strict=True)
        dummy_input = {
            "float_feature": -7168.25,
            "float_feature_2": -68.99,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0."
        with self.assertRaises(ValueError):
            serializer(
                dummy_input,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                choices_text=choices_text,
                task_context_text=task_context_text,
            )

    def test_structured_serializer_deserialize(self):
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": 7168.25,
            "int_feature": -968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0."

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices_text=choices_text,
            task_context_text=task_context_text,
        )
        feature_names = list(dummy_input.keys())
        deserialized = serializer.deserialize_example(serialized, feature_names)
        deserialized_expected = {k: str(v) for k, v in dummy_input.items()}
        self.assertDictEqual(deserialized, deserialized_expected)

    def test_structured_serializer_raises_on_deserialize_bad_example(self):
        """Test that StructuredSerializer.deserialize() raises ValueError
        on malformed example."""
        serializer = StructuredSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": 7168.25,
            "int_feature": -968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0."

        serialized = serializer(
            dummy_input,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices_text=choices_text,
            task_context_text=task_context_text,
        )
        feature_names = list(dummy_input.keys())
        for key_to_drop in (
            serializer.key_start_token,
            serializer.key_end_token,
            serializer.value_start_token,
            serializer.value_end_token,
        ):
            first_key_start = serialized.index(key_to_drop)

            # drop the first instance of token_to_drop
            bad_example = (
                serialized[:first_key_start]
                + serialized[first_key_start + len(key_to_drop) :]
            )

            with self.assertRaises(ValueError):
                serializer.deserialize_example(bad_example, feature_names=feature_names)


class TestPandasSeriesSerializer(unittest.TestCase):
    def _try_parse_serialized(self, serialized: str) -> pd.Series:
        try:
            return eval(serialized)
        except SyntaxError as se:
            raise se(f"could not evaluate serialized output: {serialized}")

    def test_pandas_serializer_no_prefix_suffix_choices(self):
        """Test PandasSeriesSerializer with only feature keys/values."""
        serializer = PandasSeriesSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        # Check that the output literally parses into the correct Pandas object.
        serialized = serializer(dummy_input, None)
        serialized_evaluated = self._try_parse_serialized(serialized)
        self.assertIsInstance(
            serialized_evaluated,
            pd.Series,
            msg=f"expected serialization to parse to pd.Series; got type {type(serialized)}",
        )
        self.assertIsInstance(
            serialized_evaluated["features"],
            dict,
            msg=f"expected 'features' to parse to dict; "
            f"got type {type(serialized_evaluated['features'])}",
        )

        expected = pd.Series(
            {"features": {k: {"value": v} for k, v in dummy_input.items()}}
        )
        pd.testing.assert_series_equal(serialized_evaluated, expected)

    def test_pandas_serializer_with_meta(self):
        """Test Pandas serializer with metadata."""
        serializer = PandasSeriesSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        meta = {
            "quantile": {"float_feature": 0.99, "int_feature": 0.91},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }

        serialized = serializer(dummy_input, meta=meta)
        serialized_evaluated = self._try_parse_serialized(serialized)

        self.assertIsInstance(
            serialized_evaluated,
            pd.Series,
            msg=f"expected serialization to parse to pd.Series; got type {type(serialized)}",
        )
        self.assertIsInstance(
            serialized_evaluated["features"],
            dict,
            msg=f"expected 'features' to parse to dict; "
            f"got type {type(serialized_evaluated['features'])}",
        )

        features_expected = {
            "features": {
                "float_feature": {
                    "value": dummy_input["float_feature"],
                    "quantile": meta["quantile"]["float_feature"],
                    "scale": meta["scale"]["float_feature"],
                },
                "int_feature": {
                    "value": dummy_input["int_feature"],
                    "quantile": meta["quantile"]["int_feature"],
                    "scale": meta["scale"]["int_feature"],
                },
                "str_feature": {
                    "value": dummy_input["str_feature"],
                },
            }
        }
        expected = pd.Series(features_expected)
        pd.testing.assert_series_equal(serialized_evaluated, expected)

    def test_pandas_serializer_with_prefix_suffix_choices(self):
        serializer = PandasSeriesSerializer(config=SerializerConfig())

        dummy_input = {
            "float_feature": -7168.25,
            "bool_feature": True,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context."
        prefix_text = "This is an observation."
        suffix_text = "What is the label?"
        choices_text = "1 or 0"

        serialized = serializer(
            dummy_input,
            task_context_text=task_context_text,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["1", "0"],
        )
        expected = pd.Series(
            {
                "task_context": task_context_text,
                "prefix": prefix_text,
                "features": {k: {"value": v} for k, v in dummy_input.items()},
                "suffix": suffix_text,
                "choices": choices_text,
            }
        )
        serialized_parsed = self._try_parse_serialized(serialized)
        pd.testing.assert_series_equal(serialized_parsed, expected)

    def test_pandas_serializer_deserialize(self):
        """Test deserialization of pandas serializer with metafeatures."""
        serializer = PandasSeriesSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -99.5,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        meta = {
            "quantile": {"float_feature": 0.99, "int_feature": 0.91},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }
        serialized = serializer(dummy_input, meta=meta)
        deserialized = serializer.deserialize_example(serialized, feature_names=None)
        expected = {
            "float_feature": {
                "value": dummy_input["float_feature"],
                "quantile": meta["quantile"]["float_feature"],
                "scale": meta["scale"]["float_feature"],
            },
            "int_feature": {
                "value": dummy_input["int_feature"],
                "quantile": meta["quantile"]["int_feature"],
                "scale": meta["scale"]["int_feature"],
            },
            "str_feature": {
                "value": dummy_input["str_feature"],
            },
        }
        self.assertDictEqual(deserialized, expected)


class TestHtmlNoWhitespaceSerializer(unittest.TestCase):
    """Simple test for HtmlNoWhitespaceSerializer.

    This class focuses on verifying that there is no whitespace between HTML tags;
    other functionality is tested in TestHtmlSerializer."""

    def test_html_no_whitespace_serializer(self):
        serializer = HtmlNoWhitespaceSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        # Check that the output literally parses into the correct Pandas object.
        serialized = serializer(dummy_input, None)
        expected = "<table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>0</th></tr></thead><tbody><tr><th>features</th><td>{'float_feature': {'value': -768.25}, 'int_feature': {'value': -25}, 'str_feature': {'value': 'my_category'}}</td></tr></tbody></table>"
        self.assertEqual(serialized, expected)
        self.assertIsNone(re.search(">\s+<", serialized))

    def test_html_no_whitespace_serializer_with_prefix_suffix_choices(self):
        serializer = HtmlNoWhitespaceSerializer(config=SerializerConfig())

        dummy_input = {
            "float_feature": -4e-3,
            "bool_feature": True,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context, which provides context."
        prefix_text = "This is an observation drawn from a dataset."
        suffix_text = "What is the label??"
        serialized = serializer(
            dummy_input,
            task_context_text=task_context_text,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["2", "1", "0"],
        )
        expected = "<table border=\"1\" class=\"dataframe\"><thead><tr style=\"text-align: right;\"><th></th><th>0</th></tr></thead><tbody><tr><th>task_context</th><td>This is the task context, which provides context.</td></tr><tr><th>prefix</th><td>This is an observation drawn from a dataset.</td></tr><tr><th>features</th><td>{'float_feature': {'value': -0.004}, 'bool_feature': {'value': True}, 'int_feature': {'value': 5968}, 'str_feature': {'value': 'my_category'}}</td></tr><tr><th>suffix</th><td>What is the label??</td></tr><tr><th>choices</th><td>2 or 1 or 0</td></tr></tbody></table>"
        self.assertEqual(serialized, expected)
        self.assertIsNone(re.search(">\s+<", serialized))


class TestHtmlSerializer(unittest.TestCase):
    def test_html_serializer(self):
        serializer = HtmlSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -768.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        # Check that the output literally parses into the correct Pandas object.
        serialized = serializer(dummy_input, None)
        expected = "<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>0</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>features</th>\n      <td>{'float_feature': {'value': -768.25}, 'int_feature': {'value': -25}, 'str_feature': {'value': 'my_category'}}</td>\n    </tr>\n  </tbody>\n</table>"

        self.assertEqual(serialized, expected)

    def test_html_serializer_with_meta(self):
        """Test HtmlSerializer with meta features."""
        serializer = HtmlSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -1e-6,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        meta = {
            "quantile": {"float_feature": 0.99, "int_feature": 0.01},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }
        serialized = serializer(dummy_input, meta=meta)
        expected = "<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>0</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>features</th>\n      <td>{'float_feature': {'value': -1e-06, 'quantile': 0.99, 'scale': -0.2}, 'int_feature': {'value': -1, 'quantile': 0.01, 'scale': -0.99}, 'str_feature': {'value': 'my_category'}}</td>\n    </tr>\n  </tbody>\n</table>"

        self.assertEqual(serialized, expected)

    def test_html_serializer_with_prefix_suffix_choices(self):
        serializer = HtmlSerializer(config=SerializerConfig())

        dummy_input = {
            "float_feature": -4e-3,
            "bool_feature": True,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context, which provides context."
        prefix_text = "This is an observation drawn from a dataset."
        suffix_text = "What is the label??"
        serialized = serializer(
            dummy_input,
            task_context_text=task_context_text,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["2", "1", "0"],
        )
        expected = "<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>0</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>task_context</th>\n      <td>This is the task context, which provides context.</td>\n    </tr>\n    <tr>\n      <th>prefix</th>\n      <td>This is an observation drawn from a dataset.</td>\n    </tr>\n    <tr>\n      <th>features</th>\n      <td>{'float_feature': {'value': -0.004}, 'bool_feature': {'value': True}, 'int_feature': {'value': 5968}, 'str_feature': {'value': 'my_category'}}</td>\n    </tr>\n    <tr>\n      <th>suffix</th>\n      <td>What is the label??</td>\n    </tr>\n    <tr>\n      <th>choices</th>\n      <td>2 or 1 or 0</td>\n    </tr>\n  </tbody>\n</table>"
        self.assertEqual(serialized, expected)


class TestJsonSerializer(unittest.TestCase):
    def test_json_serializer(self):
        serializer = JsonSerializer(config=SerializerConfig())
        dummy_input = {
            "a_float_feature": -0.25,
            "int_feature": -25,
            "str_feature": "my_category",
        }
        # Check that the output literally parses into the correct Pandas object.
        serialized = serializer(dummy_input, None)
        expected = json.dumps(
            {"features": {k: {"value": v} for k, v in dummy_input.items()}}
        )
        self.assertEqual(serialized, expected)

    def test_json_serializer_with_meta(self):
        """Test JsonSerializer with meta features."""
        serializer = JsonSerializer(config=SerializerConfig())
        dummy_input = {
            "float_feature": -1e-6,
            "int_feature": -1,
            "str_feature": "my_category",
        }
        meta = {
            "quantile": {"float_feature": 99e-4, "int_feature": 0.01},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }
        serialized = serializer(dummy_input, meta=meta)
        expected = '{"features": {"float_feature": {"value": -1e-06, "quantile": 0.0099, "scale": -0.2}, "int_feature": {"value": -1, "quantile": 0.01, "scale": -0.99}, "str_feature": {"value": "my_category"}}}'
        self.assertEqual(serialized, expected)

    def test_json_serializer_with_prefix_suffix_choices(self):
        serializer = JsonSerializer(config=SerializerConfig())

        dummy_input = {
            "float_feature": -4e-3,
            "bool_feature": True,
            "int_feature": 5968,
            "str_feature": "my_category",
        }
        task_context_text = "This is the task context, which provides context."
        prefix_text = "This is an observation drawn from a dataset."
        suffix_text = "What is the label??"
        serialized = serializer(
            dummy_input,
            task_context_text=task_context_text,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["2", "1", "0"],
        )
        expected = '{"task_context": "This is the task context, which provides context.", "prefix": "This is an observation drawn from a dataset.", "features": {"float_feature": {"value": -0.004}, "bool_feature": {"value": true}, "int_feature": {"value": 5968}, "str_feature": {"value": "my_category"}}, "suffix": "What is the label??", "choices": ["2", "1", "0"]}'
        self.assertEqual(serialized, expected)

    def test_json_serializer_with_prefix_suffix_choices_meta(self):
        serializer = JsonSerializer(config=SerializerConfig())

        dummy_input = {
            "float_feature": -4e-3,
            "bool_feature": True,
            "int_feature": 5968,
            "str_feature": "my_category",
        }

        meta = {
            "quantile": {"float_feature": 99e-4, "int_feature": 0.01},
            "scale": {"float_feature": -0.2, "int_feature": -0.99},
        }
        task_context_text = "This is the task context, which provides context."
        prefix_text = "This is an observation drawn from a dataset."
        suffix_text = "What is the label??"
        serialized = serializer(
            dummy_input,
            meta=meta,
            task_context_text=task_context_text,
            prefix_text=prefix_text,
            suffix_text=suffix_text,
            choices=["2", "1", "0"],
        )
        expected = '{"task_context": "This is the task context, which provides context.", "prefix": "This is an observation drawn from a dataset.", "features": {"float_feature": {"value": -0.004, "quantile": 0.0099, "scale": -0.2}, "bool_feature": {"value": true}, "int_feature": {"value": 5968, "quantile": 0.01, "scale": -0.99}, "str_feature": {"value": "my_category"}}, "suffix": "What is the label??", "choices": ["2", "1", "0"]}'

        self.assertEqual(serialized, expected)
