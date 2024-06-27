"""
Tests for inference.

To run tests: python -m unittest rtfm/tests/test_inference.py -v
"""
import unittest
from dataclasses import dataclass
from typing import List

import einops
import numpy as np
import pandas as pd
import torch
from llama_recipes.inference.model_utils import load_model
from tqdm import tqdm
from transformers import AutoTokenizer

from rtfm.configs import TrainConfig, TokenizerConfig
from rtfm.inference_utils import infer_on_example
from rtfm.serialization.serializers import get_serializer
from rtfm.special_tokens import EOC_TOKEN
from rtfm.tokenization.text import prepare_tokenizer


@dataclass
class FixedResponseDummyModel:
    """A dummy model for testing purposes."""

    response_tokens: torch.Tensor

    def generate(self, input_ids: torch.Tensor, *args, **kwargs) -> List[torch.Tensor]:
        return [
            torch.cat(
                (input_ids, self.response_tokens.to(input_ids.device).view(1, -1)),
                dim=1,
            ).flatten()
        ]


class TestInferenceTinyLlama(unittest.TestCase):
    """Test inference with a tiny llama model.

    This is an integration test - it will only verify that no exceptions/errors are raised.
    """

    def setUp(self):
        train_config = TrainConfig(
            model_name="yujiepan/llama-2-tiny-random", context_length=4096
        )
        tokenizer_config = TokenizerConfig()
        model = load_model(
            train_config.model_name,
            quantization=False,
            use_fast_kernels=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
        serializer = get_serializer(train_config.serializer_cls)
        tokenizer, model = prepare_tokenizer(
            model,
            tokenizer=tokenizer,
            pretrained_model_name_or_path=train_config.model_name,
            model_max_length=train_config.context_length,
            use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
            serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
            serializer_tokens=serializer.special_tokens
            if tokenizer_config.add_serializer_tokens
            else None,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.serializer = serializer

    def test_inference_few_shot_with_labeled_example(self):
        """Test inference with tiny llama 2 model (few-shot), where the target example is labeled."""
        labeled_examples = pd.DataFrame(
            [{"X1": 1, "X2": 0, "y": 0}, {"X1": 1, "X2": 1, "y": 1}]
        )
        target_example = pd.DataFrame(
            [
                {"X1": 1, "X2": 0, "y": 1},
            ]
        )
        _ = infer_on_example(
            model=self.model,
            tokenizer=self.tokenizer,
            serializer=self.serializer,
            target_colname="y",
            target_choices=["0", "1"],
            target_example=target_example,
            labeled_examples=labeled_examples,
            handle_invalid_predictions="warn",
        )

    def test_inference_few_shot(self):
        """Test inference with tiny llama 2 model (few-shot),
        where the target example does not contain the target column."""
        labeled_examples = pd.DataFrame(
            [{"X1": 1, "X2": 0, "y": 0}, {"X1": 1, "X2": 1, "y": 1}]
        )
        target_example = pd.DataFrame(
            [
                {"X1": 1, "X2": 0},
            ]
        )
        _ = infer_on_example(
            model=self.model,
            tokenizer=self.tokenizer,
            serializer=self.serializer,
            target_colname="y",
            target_choices=["0", "1"],
            target_example=target_example,
            labeled_examples=labeled_examples,
            handle_invalid_predictions="warn",
        )

    def test_inference_zero_shot(self):
        """Test inference with tiny llama 2 model (zero-shot)."""
        target_example = pd.DataFrame(
            [
                {
                    "X1": 1,
                    "X2": 0,
                },
            ]
        )
        output = infer_on_example(
            model=self.model,
            tokenizer=self.tokenizer,
            serializer=self.serializer,
            target_colname="y",
            target_choices=["0", "1"],
            target_example=target_example,
            labeled_examples=None,
            handle_invalid_predictions="warn",
        )


class TestInferenceDummyModel(unittest.TestCase):
    """Test inference with a dummy model.

    This class effectively tests that the inference loop returns the correct labels
    and predictions, and serves as a further integration test."""

    def setUp(self):
        train_config = TrainConfig(
            model_name="yujiepan/llama-2-tiny-random", context_length=4096
        )
        tokenizer_config = TokenizerConfig()
        model = load_model(
            train_config.model_name,
            quantization=False,
            use_fast_kernels=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
        serializer = get_serializer(train_config.serializer_cls)
        tokenizer, _ = prepare_tokenizer(
            model,
            tokenizer=tokenizer,
            pretrained_model_name_or_path=train_config.model_name,
            model_max_length=train_config.context_length,
            use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
            serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
            serializer_tokens=serializer.special_tokens
            if tokenizer_config.add_serializer_tokens
            else None,
        )
        self.tokenizer = tokenizer

        response_tokens = torch.Tensor(tokenizer("0.0")["input_ids"][1:]).long()
        response_tokens = torch.concat(
            (
                response_tokens,
                torch.Tensor(tokenizer(EOC_TOKEN)["input_ids"])[1:].long(),
            )
        )
        self.model = FixedResponseDummyModel(response_tokens)
        self.serializer = serializer

    def test_inference_few_shot(self, num_shots=10, num_iters=512):
        """Test inference with tiny llama 2 model (few-shot)."""
        df = pd.read_csv(
            "sampledata/dummy_n10000_d4_numclasses4/dummy_n10000_d4_numclasses4.csv"
        )
        preds = []
        labels = []

        # iterate over random data and verify the output matches expected proportion.
        for i in tqdm(range(num_iters), desc="infer with dummy model", total=num_iters):
            labeled_examples = df.drop([i]).sample(num_shots)
            target_example = pd.DataFrame([df.iloc[i]])

            target_colname = "target"
            output = infer_on_example(
                model=self.model,
                tokenizer=self.tokenizer,
                serializer=self.serializer,
                target_colname=target_colname,
                target_choices=df[target_colname].unique().tolist(),
                target_example=target_example,
                labeled_examples=labeled_examples,
                handle_invalid_predictions="warn",
            )
            preds.append(output)
            labels.append(target_example[target_colname].item())
        print(np.mean([x == str(y) for x, y in zip(preds, labels)]))
        np.testing.assert_allclose(
            np.mean([x == str(y) for x, y in zip(preds, labels)]),
            0.25,
            atol=0.01,
            rtol=0.04,
        )
