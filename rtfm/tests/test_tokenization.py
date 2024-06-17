"""
Tests for batch tokenization.

To run tests: python -m unittest rtfm/tests/test_tokenization.py -v
"""
import os
import random
import unittest

import numpy as np
import scipy
import torch
import transformers

from rtfm.arguments import ModelArguments, DataArguments
from rtfm.configs import TrainConfig
from rtfm.data import tokenize_batch, example_ids_to_attention_mask
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer


class TestTokenizeBatch(unittest.TestCase):
    """Tests for tokenization of a batch."""

    def setUp(self) -> None:
        self.inputs = [
            "This is a sample input. Is it red? Yes or no.",
            "This is another sample input. Is it red? Yes or no.",
        ]
        self.targets = ["Yes", "No"]
        self.model_arguments = ModelArguments(
            model_name_or_path="yujiepan/llama-2-tiny-random",
            serializer_cls="BasicSerializer",
        )
        self.training_arguments = TrainConfig(context_length=4096)
        self.data_arguments = DataArguments()
        serializer = get_serializer(self.model_arguments.serializer_cls)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_arguments.model_name_or_path,
            torch_dtype=torch.float32,
            cache_dir=None,
        )

        self.tokenizer, self.model = prepare_tokenizer(
            model,
            self.model_arguments.model_name_or_path,
            use_fast_tokenizer=self.model_arguments.use_fast_tokenizer,
            serializer_tokens_embed_fn=self.model_arguments.serializer_tokens_embed_fn,
            serializer_tokens=serializer.special_tokens,
        )

    def test_tokenize_train_batch(self):
        """Integration test to make sure tokenization works with normal inputs."""
        batch = {"input_text": self.inputs, "target_text": self.targets}
        outputs = tokenize_batch(
            batch,
            tokenizer=self.tokenizer,
            data_arguments=self.data_arguments,
            is_train=True,
        )
        self.assertTrue(
            all(len(outputs[k]) == len(self.inputs) for k in outputs.keys())
        )

    def _test_tokenize_batch_size(self, batch_size: int):
        """Integration test to make sure tokenization works with a fixed batch size."""
        batch = {
            "input_text": [random.choice(self.inputs) for _ in range(batch_size)],
            "target_text": [random.choice(self.targets) for _ in range(batch_size)],
        }

        outputs = tokenize_batch(
            batch,
            tokenizer=self.tokenizer,
            data_arguments=self.data_arguments,
            is_train=True,
        )

        self.assertTrue(all(len(outputs[k]) == batch_size for k in outputs.keys()))

    def test_tokenize_large_batch(self):
        """Integration test to make sure tokenization works with large batches (the default HF batch size of 1000)."""
        self._test_tokenize_batch_size(1000)

    def test_tokenize_single_element_batch(self):
        """Integration test to make sure tokenization works with batch size of 1."""
        self._test_tokenize_batch_size(1)


class TestPrepareTokenizer(unittest.TestCase):
    def _test_prepare_tokenizer_special_tokens(self, serializer_cls: str):
        """Generic method to test that special tokens are added for various serializers."""
        model_arguments = ModelArguments(
            model_name_or_path="yujiepan/llama-2-tiny-random",
            serializer_cls=serializer_cls,
        )
        training_arguments = TrainConfig(model_max_length=4096, output_dir="")
        serializer = get_serializer(model_arguments.serializer_cls)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_arguments.model_name_or_path,
            torch_dtype=torch.float32,
            cache_dir=None,
        )

        # For this test, we use the Llama2 tokenizer, because the "tiny"
        # tokenizer associated with the tiny-random model cannot be modified.
        assert (
            os.environ["HUGGING_FACE_HUB_TOKEN"] or os.environ["HF_TOKEN"]
        ), "either HUGGING_FACE_HUB_TOKEN or HF_TOKEN must be set."

        tokenizer, model = prepare_tokenizer(
            model,
            model_arguments.model_name_or_path,
            training_arguments,
            use_fast_tokenizer=model_arguments.use_fast_tokenizer,
            serializer_tokens_embed_fn=model_arguments.serializer_tokens_embed_fn,
            serializer_tokens=serializer.special_tokens,
        )

        # Check that all special tokens are added
        for token_name, token in serializer.special_tokens.items():
            self.assertIn(token, tokenizer.get_vocab())

    def test_prepare_tokenizer_special_tokens_basic_serializer(self):
        """Check that all special tokens are added to the vocab when using BasicSerializer."""
        self._test_prepare_tokenizer_special_tokens(serializer_cls="BasicSerializer")

    def test_prepare_tokenizer_special_tokens_structured_serializer(self):
        """Check that all special tokens are added to the vocab when using StructuredSerializer."""
        self._test_prepare_tokenizer_special_tokens(
            serializer_cls="StructuredSerializer"
        )


class Test4DAttentionMask(unittest.TestCase):
    def test_expand_attention_mask_single_example(self, seq_len=7):
        """Test example_ids_to_attention_mask with a single element containing a single example."""
        example_ids = [0] * seq_len
        mask = example_ids_to_attention_mask(example_ids)
        expected = np.tril(np.ones((seq_len, seq_len)))
        np.testing.assert_array_equal(mask, expected)

    def test_expand_attention_mask_four_examples(self, seq_lens=(7, 9, 3, 5)):
        """Test example_ids_to_attention_mask with a single element containing four examples."""
        example_ids = [
            idx for i, seq_len in enumerate(seq_lens) for idx in [i] * seq_len
        ]

        blocks = [np.tril(np.full((x, x), True)) for x in seq_lens]
        expected = scipy.linalg.block_diag(*blocks)

        mask = example_ids_to_attention_mask(example_ids)
        np.testing.assert_array_equal(mask, expected)
