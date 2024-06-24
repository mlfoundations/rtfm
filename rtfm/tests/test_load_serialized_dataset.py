"""
Tests for loading of serialized datasets.

To run tests: python -m unittest rtfm/tests/test_load_serialized_dataset.py -v

"""

import unittest

from transformers import AutoModelForCausalLM

from rtfm.arguments import ModelArguments, DataArguments
from rtfm.configs import TrainConfig
from rtfm.data import load_tokenize_and_serialize_tabular_dataset
from rtfm.serialization import BasicSerializer
from rtfm.tokenization.text import prepare_tokenizer


class TestLoadSerializedDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_args = DataArguments()
        self.model_args = ModelArguments(
            model_name_or_path="yujiepan/llama-2-tiny-random"
        )
        self.training_args = TrainConfig(output_dir=".")
        self.serializer = BasicSerializer()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path
        )
        self.tokenizer, self.model = prepare_tokenizer(
            self.model,
            self.model_args.model_name_or_path,
            4096,
            self.model_args.use_fast_tokenizer,
            self.model_args.serializer_tokens_embed_fn,
            self.serializer.special_tokens,
        )

    def test_load_adult(self):
        """Integration test to verify adult dataset loads."""
        ds_dict = load_tokenize_and_serialize_tabular_dataset(
            self.tokenizer,
            ["adult"],
            data_arguments=self.data_args,
            serializer=self.serializer,
        )
        for split, ds in ds_dict.items():
            batch = next(iter(ds))
            self.assertIsNotNone(batch, f"batch is none for split {split}")

    def test_load_multi_task(self):
        """Integration test to verify a multi-task dataset loads."""
        ds_dict = load_tokenize_and_serialize_tabular_dataset(
            self.tokenizer,
            [
                "adult",
                "cars",
            ],
            data_arguments=self.data_args,
            serializer=self.serializer,
        )
        for split, ds in ds_dict.items():
            batch = next(iter(ds))
            self.assertIsNotNone(batch, f"batch is none for split {split}")

    def test_load_max_samples_too_large(self):
        """Test the case where max_samples > dataset size."""
        ds_dict = load_tokenize_and_serialize_tabular_dataset(
            self.tokenizer,
            ["cars"],
            data_arguments=self.data_args,
            serializer=self.serializer,
            as_iterable=False,
            max_samples=int(1e6),
        )
        for split, ds in ds_dict.items():
            batch = next(iter(ds))
            self.assertIsNotNone(batch, f"batch is none for split {split}")
