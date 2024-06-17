"""
To run tests: python -m unittest rtfm/tests/test_preprocess.py -v
"""
import unittest

import transformers

from rtfm.arguments import DataArguments
from rtfm.data import (
    add_qa_and_eoc_tokens_to_example,
    tokenize_batch,
)
from rtfm.special_tokens import QA_SEP_TOKEN, EOC_TOKEN
from rtfm.tokenization.text import prepare_tokenizer, unmasked_token_idxs


class TestPreprocess(unittest.TestCase):
    def test_train_inputs_and_labels(
        self, model_name_or_path="yujiepan/llama-2-tiny-random"
    ):
        """Test that detokenized inputs match original inputs + QA_SEP+TOKEN.

        This function is also useful as it tests the most basic version
        of our input processing pipeline."""

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer, model = prepare_tokenizer(
            model,
            model_name_or_path,
            use_fast_tokenizer=False,
            serializer_tokens_embed_fn="smart",
            serializer_tokens=None,
        )

        inputs = [
            "This is a sample input. Is it red? Yes or no.",
            "This is another sample input. Is it red? Yes or no.",
        ]
        targets = ["Yes", "No"]

        ## The following lines mimic the pipeline in load_serialized_tabular_dataset()

        # This mimics the outputs of example_map_fn
        examples = [
            {"text": x, "class_label_as_text": y} for x, y in zip(inputs, targets)
        ]

        examples = [add_qa_and_eoc_tokens_to_example(x) for x in examples]

        # Convert to batch format to mimic HF batch formatting
        batch = {
            k: [example[k] for example in examples]
            for k in ["input_text", "target_text"]
        }
        preprocessed = tokenize_batch(
            batch, tokenizer, data_arguments=DataArguments(), is_train=True
        )

        # Now that inputs have been preprocessed, we detokenize
        # to verify that the preprocessing produces examples that
        # are formatted as expected.
        preprocessed_detokenized = tokenizer.batch_decode(
            preprocessed["input_ids"], skip_special_tokens=True
        )

        # Check that inputs have form '<INPUT TEXT><QA_SEP_TOKEN><Target><EOC_TOKEN>'
        # e.g. "This is a sample input. Is it red? Yes or no.{QA_SEP_TOKEN}Yes{EOC_TOKEN}"
        for input, target, preprocessed_input in zip(
            inputs, targets, preprocessed_detokenized
        ):
            input_roundtrip = "".join(preprocessed_input).strip()
            full_input = input + QA_SEP_TOKEN + target + EOC_TOKEN
            self.assertEqual(full_input, input_roundtrip)

        # Check that targets have form '<TARGET><ENDOFCHUNK>', e.g. 'Yes<|endcompletion|>'.
        for target, preprocessed_target in zip(targets, preprocessed["labels"]):
            target_and_eoc = target + EOC_TOKEN

            idxs = unmasked_token_idxs(preprocessed_target)
            target_detokenized = tokenizer.decode(preprocessed_target[idxs])
            target_roundtrip = "".join(target_detokenized).strip()
            self.assertEqual(target_and_eoc, target_roundtrip)

        # Check that inputs and targets tokens have same shape
        for x, y in zip(preprocessed["input_ids"], preprocessed["labels"]):
            self.assertEqual(len(x), len(y))
