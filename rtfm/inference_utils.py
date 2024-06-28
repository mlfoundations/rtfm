import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Literal, Any

import numpy as np
import pandas as pd
import transformers
from datasets.utils.logging import (
    disable_progress_bar,
    enable_progress_bar,
)

disable_progress_bar()

from rtfm.arguments import DataArguments
from rtfm.data import (
    serialize_dataset_fn,
    add_qa_and_eoc_tokens_to_example,
    tokenize_ds_dict,
    build_formatted_df,
    prepare_hf_dataset_from_formatted_df,
    make_few_shot_sample,
    DataCollatorForSupervisedDataset,
)
from rtfm.generation_utils import (
    make_eoc_stopping_criterion,
    prepare_input_ids_and_attention_mask_for_generation,
    parse_generated_text,
)
from rtfm.serialization.serializers import RowSerializer
from rtfm.task_config import TLMConfig
from rtfm.torch_utils import batch_to_xpu


def prepare_dataframe(
    df, target_colname: str, data_arguments: DataArguments, as_iterable: bool = False
):
    # TODO(jpgard): construct the info in a way that matches load_uncached_hf_dataset().
    ds_info = json.dumps({"task": "dummy_task", "target": target_colname})
    df = build_formatted_df(df, ds_info, data_arguments)
    return prepare_hf_dataset_from_formatted_df(df, as_iterable=as_iterable)


def infer_on_example(
    model: transformers.AutoModelForCausalLM,
    tokenizer,
    serializer: RowSerializer,
    target_example: pd.DataFrame,
    target_colname: str,
    target_choices: List[Any],
    labeled_examples: Optional[pd.DataFrame] = None,
    max_new_tokens: Optional[int] = None,
    cfg: Optional[TLMConfig] = None,
    handle_invalid_predictions: Literal["raise", "warn", None] = "raise",
) -> str:
    """

    :param model: the model to perform inference with.
    :param tokenizer: the tokenizer to use. Note that if you are using a base model
    (i.e. the Llama 3-8B), you should NOT add special tokens to the tokenizer,
    as these need to be learned during fine-tuning (set add_special_tokens=False
    in rtfm.tokenization.text.prepare_tokenizer()).
    :param serializer: the serializer to use. This should be the same serializer
    used when fine-tuning the model. We recommend the use of BasicSerializerV2,
    as this is the serializer used for all of our public releases (i.e. TabuLa-8B).
    :param target_example: a single-row DataFrame to predict on. This DataFrame can
    optionally include the target column (in which case it will be ignored).
    :param target_colname: the name of the target column to predict. This column should
    be present in the few-shot examples.
    :param target_choices: a list of potential choices for the target column. The objects
    can be of any type castable to string. For the most reliable performance we suggest
    explicitly casting to strings.
    :param labeled_examples: DataFrame of "shots" to use when making predictions.
    These examples will be included in the exact order they are provided.
    :param max_new_tokens: maximum number of new tokens to generate. Note that this
    should be enough to cover any of the potential target classes, and the <|endcompletion|>
    token. If None, no limit is enforced. Note that generation automatically terminates
    once the <|endcompletion|> token is generated.
    :param cfg: an optional TLMConfig object. Use this to provide finer-grained control
    over the serialization.
    :param handle_invalid_predictions: How to handle invalid predictions from the model.
    If "raise", an exception is raised; if "warn", a warning is logged; if None
    the function returns the invalid completion as if it were a valid completion.
    :return: a string containing the prediction.
    """
    is_fewshot = labeled_examples is not None
    disable_progress_bar()

    if len(target_example) != 1:
        raise ValueError("Only use one example at a time for inference.")
    if target_colname not in target_example.columns:
        logging.warning(
            f"Column {target_colname} is not in target example; "
            f"got columns {target_example.columns}. Adding a dummy placeholder "
            "with empty values for preprocessing. This behavior is expected if your "
            "target samples do not contain the target column at all."
        )
        target_example[target_colname] = np.nan
    else:
        target_example.pop(target_colname)

    if is_fewshot:
        assert target_colname in labeled_examples.columns, (
            f"Expected column {target_colname} "
            f"in labeled examples; got columns {target_example.columns}"
        )
    data_arguments = DataArguments(
        use_config=True,
        feature_value_handling="none",
        feature_name_handling="none",
        targets_handling="none",
    )

    if cfg is None:
        cfg = TLMConfig(
            prefix=f"Predict the {target_colname}",
            suffix=f"What is the value of {target_colname}?",
            label_values=target_choices,
        )

    # Build datasets from the examples, to follow the same procedure
    # used for training + evaluation.
    if is_fewshot:
        ds_dict = {
            "train": prepare_dataframe(
                labeled_examples, target_colname, data_arguments
            ),
            "test": prepare_dataframe(target_example, target_colname, data_arguments),
        }
    else:
        ds_dict = {
            "test": prepare_dataframe(target_example, target_colname, data_arguments),
        }

    ds_dict = {
        split: serialize_dataset_fn(
            ds, data_args=data_arguments, serializer=serializer, cfg=cfg
        )
        for split, ds in ds_dict.items()
    }
    ds_dict = {
        split: ds.map(add_qa_and_eoc_tokens_to_example) for split, ds in ds_dict.items()
    }
    # At this point, the datasets have the same format as the output of
    # rtfm.data.load_serialized_dataset().

    tokenized_ds_dict = tokenize_ds_dict(
        ds_dict, tokenizer=tokenizer, data_arguments=data_arguments
    )

    # Shots is a list of dictionaries
    if is_fewshot:
        ds_train = tokenized_ds_dict["train"]
        shots = list(
            ds_train.select_columns(["input_ids", "labels"])
            .with_format("torch")
            .take(len(labeled_examples))
        )
        # Make shots a list of tuples
        shots = [(x["input_ids"], x["labels"]) for x in shots]
    else:
        shots = None

    # For the target_sample, we still call list() even though it is one
    # element in order to actually trigger the lazy-loading of the data.
    ds_test = tokenized_ds_dict["test"]
    target_sample = list(
        ds_test.select_columns(["input_ids", "labels"]).with_format("torch").take(1)
    )[0]

    target_sample = (target_sample["input_ids"], target_sample["labels"])

    enable_progress_bar()

    # Make the few-shot example.
    input_ids, labels = make_few_shot_sample(
        shots=shots, target_sample=target_sample, max_len=tokenizer.model_max_length
    )

    # Set up dataloader for inference
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    batch = data_collator([{"input_ids": input_ids, "labels": labels}])

    # Handle new (i.e. generated) token limit
    available_context_window_tokens = (
        tokenizer.model_max_length - batch["input_ids"].numel()
    )

    if available_context_window_tokens < 2:
        raise ValueError(
            f"Not enough remaining context tokens to generate a valid output. "
            f"After serialization, the input example contains {batch['input_ids'].numel()} tokens."
        )

    if max_new_tokens:
        if max_new_tokens > available_context_window_tokens:
            logging.warning(
                f"max_new_tokens is {max_new_tokens} but after serialization, "
                f"the input example contains {batch['input_ids'].numel()} tokens which"
                f"only leaves space for {available_context_window_tokens} "
                f"with context window size {tokenizer.model_max_length}."
            )
        max_new_tokens = min(available_context_window_tokens, max_new_tokens)
    else:
        max_new_tokens = available_context_window_tokens

    # Prepare batch on device and create inputs + attention mask
    batch = batch_to_xpu(batch)
    input_ids, attention_mask = prepare_input_ids_and_attention_mask_for_generation(
        batch
    )

    # Generate and decode
    stopping_criterion = make_eoc_stopping_criterion(input_ids, tokenizer)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        stopping_criteria=[stopping_criterion],
    )
    decoded_text = tokenizer.batch_decode(output)[0].strip()
    prediction_text, is_valid = parse_generated_text(decoded_text)

    # Exception handling
    if is_valid or handle_invalid_predictions is None:
        return prediction_text
    elif handle_invalid_predictions == "raise":
        raise ValueError(
            f"Model returned invalid predictions (no EOC token): {prediction_text}"
        )
    elif handle_invalid_predictions == "warn":
        logging.warning(
            "model returned invalid text (no EOC token); returning incomplete prediction text"
        )
        return prediction_text
    else:
        raise ValueError(
            f"unknown value for handle_invalid_predictions: {handle_invalid_predictions}"
        )


@dataclass
class InferenceModel:
    """Wrapper to support easy inference.

    This class holds a model, tokenizer, and serializer, and uses them for
    inference on tabular data.

    Note that this is a convenience wrapper around infer_on_example(); if you prefer
    the same functionality can be achieved by calling infer_on_example() directly
    and passing the model, tokenizer, and serializer as additional parameters
    to infer_on_example()."""

    model: transformers.AutoModelForCausalLM
    tokenizer: transformers.PreTrainedTokenizer
    serializer: RowSerializer

    def predict(
        self,
        target_example: pd.DataFrame,
        target_colname: str,
        target_choices: List[str],
        **kwargs,
    ):
        return infer_on_example(
            model=self.model,
            tokenizer=self.tokenizer,
            serializer=self.serializer,
            target_example=target_example,
            target_colname=target_colname,
            target_choices=target_choices,
            **kwargs,
        )
