import copy
import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, Sequence, List, Union, Optional, Any, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import transformers
import webdataset as wds

import datasets
from datasets import load_dataset, IterableDataset, Dataset
from rtfm.arguments import DataArguments
from rtfm.configs import TrainConfig
from rtfm.datasets import get_task_dataset
from rtfm.datasets.data_utils import (
    make_object_json_serializable,
    is_date_column,
    df_to_records,
    build_formatted_df,
)
from rtfm.datasets.tableshift_utils import (
    fetch_preprocessor_config_from_data_args,
    get_dataset_info,
)
from rtfm.serialization.serializers import RowSerializer
from rtfm.special_tokens import IGNORE_INDEX, QA_SEP_TOKEN, EOC_TOKEN
from rtfm.task_config import (
    get_tlm_config,
    TLMConfig,
)

_HF_VERSION_MAJOR, _HF_VERSION_MINOR, _ = [
    int(x) for x in transformers.__version__.split(".")
]


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    padding="longest",
    truncation="do_not_truncate",
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=truncation,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_and_targets = [s + t for s, t in zip(sources, targets)]
    # At eval time we pad to max_length to ensure batches have same shape across devices
    padding = "longest"
    sources_and_targets_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, padding=padding)
        for strings in (sources_and_targets, sources)
    ]

    input_ids = sources_and_targets_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def example_ids_to_attention_mask(example_ids: List[int]) -> np.ndarray:
    """Construct a boolean attention mask from a sequence of example_ids, representing a single element in a batch.

    :param example_ids: List of length (seq_len) containing token IDs as integers.

    The output is a np.array of type bool, with lower-block-triangular entries.
    """
    assert isinstance(
        example_ids, list
    ), f"expected list of example_ids, got type {type(example_ids)}"
    max_example_id = max(example_ids)
    example_ids = torch.Tensor(example_ids)
    block_sizes = [(example_ids == i).sum() for i in range(max_example_id + 1)]
    blocks = [np.tril(np.full((x, x), True)) for x in block_sizes]
    mask = scipy.linalg.block_diag(*blocks)
    return mask


def prepare_4d_attention_mask(instances: Sequence[Dict]) -> np.ndarray:
    # each attention mask is of shape [seq_len, seq_len]
    attention_masks = [
        example_ids_to_attention_mask(x["example_ids"]) for x in instances
    ]
    try:
        attention_mask = np.stack(
            attention_masks, axis=0
        )  # shape [batch_size, seq_len, seq_len]
    except ValueError as ve:
        if "all input arrays must have the same shape" in str(ve):
            logging.warning(
                "ValueError in prepare_4d_attention_mask(); expected all attention "
                f"masks to have same shape; got shapes {[x.shape for x in attention_masks]}"
            )
            raise ve
    attention_mask = np.expand_dims(
        attention_mask, axis=1
    )  # shape [batch_size, 1, seq_len, seq_len]
    if _HF_VERSION_MAJOR == 4 and _HF_VERSION_MINOR <= 41:
        return attention_mask
    else:
        return 1 - attention_mask


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_position_ids: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        output = dict(input_ids=input_ids, labels=labels)

        if any("example_ids" in x for x in instances):
            # Case: this is a packed batch. Prepare the 4d attention mask, and the position IDs.
            attention_mask = prepare_4d_attention_mask(instances)
            attention_mask = torch.from_numpy(attention_mask).to(input_ids.device)
            output.update(dict(attention_mask=attention_mask))
            if self.use_position_ids:
                position_ids = torch.LongTensor(
                    [instance["position_ids"] for instance in instances]
                )
                output.update(dict(position_ids=position_ids))

        else:
            # Case: this is not a packed batch. Prepare a standard attention mask.
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            output.update(dict(attention_mask=attention_mask))
        return output


def make_hf_name(name: str) -> str:
    for char in "<>:/\|?*":
        if char in name:
            name = name.replace(char, "_")
    return name


class NoTargetCandidatesError(ValueError):
    """Raised when there are no valid targets in a dataframe."""

    pass


class DatasetTypeError(TypeError):
    """Raised when there is a TypeError dumping a dataset element to JSON."""

    pass


def is_numeric(s) -> bool:
    """Check whether a string is numeric. This includes floats such as '3.5' and 3.'."""
    return bool(re.match(r"^-?\d+(\.+\d+)?$", s))


def is_numeric_series(vals: Union[pd.Series, Sequence[str]]) -> bool:
    return all(is_numeric(x) for x in vals)


def is_valid_target_column(
    data_args: DataArguments, ser: pd.Series, unique_values_serializable: Sequence[str]
) -> bool:
    """Check whether a target column is valid based on data_args."""
    if "Unnamed:" in ser.name:
        logging.warning(f"excluding target candidate {ser.name}")
        return False

    if data_args.labels_drop_dates and is_date_column(ser):
        logging.warning(
            f"excluding target candidate {ser.name} due to being of date type {ser.dtype}."
        )
        return False

    if ser.nunique() < data_args.labels_min_unique_values:
        logging.warning(
            f"excluding target candidate {ser.name} due to "
            f"insufficient number of unique values ({ser.nunique()} < data_args.labels_min_unique_values)"
        )
        return False

    all_values_are_numeric = is_numeric_series(unique_values_serializable)
    if (
        data_args.labels_require_nonunique
        and ser.nunique() == len(ser)
        # Allow numeric columns to have all unique values if labels_drop_numeric is False.
        and (not data_args.labels_drop_numeric and all_values_are_numeric)
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to only unique values"
        )
        return False

    if (
        data_args.labels_drop_numeric
        and all_values_are_numeric
        # Allow numeric columns if they are binary {0,1}.
        and not set(unique_values_serializable) == {"0", "1"}
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to being of numeric type"
        )
        return False

    if any(
        len(str(x)) > data_args.max_target_len_chars for x in unique_values_serializable
    ):
        logging.warning(
            f"excluding target candidate {ser.name} due to values exceeding {data_args.max_target_len_chars} chars"
        )
        return False
    return True


def build_formatted_df_from_file(file, data_args: DataArguments) -> pd.DataFrame:
    """Build a formatted DataFrame.

    The result of this function has columns 'data' and 'info', which are used for
    downstream processing.
    """
    assert not data_args.use_metafeatures

    # Read the data
    if file.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.endswith(".parquet"):
        df = pd.read_parquet(file)
    else:
        raise ValueError(f"unknown file format: {file}")

    # Iterate over the columns in the dataframe, and if it is a valid
    # candidate for being a target (more than one distinct value)
    # then add it to target_candidates_and_unique_values(). After this loop, target_candidates_and_unique_values
    # contains all target candidates.
    target_candidates_and_unique_values: Dict[str, pd.Series] = {}
    for c in df.columns:
        try:
            # Check that the values of the target column are not too long.
            unique_values_serializable = (
                df[c].apply(make_object_json_serializable).unique()
            )

            if not is_valid_target_column(data_args, df[c], unique_values_serializable):
                continue
            else:
                target_candidates_and_unique_values[c] = unique_values_serializable

        except TypeError as te:
            # Case: there is an unhashable type in the targets, so it cannot
            # be counted with pd.Series.unique(); we do not consider these
            # as potential candidates to avoid typecasting the column.
            if "unhashable type" in str(te):
                continue
            else:
                raise te

    # Compute weighted probabilities for the target candidates.
    target_candidates = list(target_candidates_and_unique_values.keys())
    numeric_count = sum(
        is_numeric_series(vals) for vals in target_candidates_and_unique_values.values()
    )
    nonnumeric_count = len(target_candidates) - numeric_count

    p = data_args.labels_p_numeric
    target_probs = (
        [
            p / numeric_count if is_numeric_series(vals) else (1 - p) / nonnumeric_count
            for vals in target_candidates_and_unique_values.values()
        ]
        if numeric_count and nonnumeric_count
        else None
    )
    if target_probs:
        target_probs = np.array(target_probs) / sum(target_probs)

    if not target_candidates:
        raise NoTargetCandidatesError

    # Choose a target uniformly at random for the table. This target will be used for all examples in the table.
    target = np.random.choice(target_candidates, p=target_probs)

    if all(is_numeric(x) for x in target_candidates_and_unique_values[target]):
        num_buckets = np.random.randint(2, 9)
        # case: this is a continuous column; discretize it.
        from rtfm.serialization.serialization_utils import discretize_continuous_column

        # TODO(jpgard): currently the below will fail with an AssertionError
        #  for columns where is_numeric(x) is True but pd.api.types.is_numeric_dtype(x) is False.
        df[target] = discretize_continuous_column(df[target], num_buckets=num_buckets)
        target_candidates_and_unique_values[target] = (
            df[target].apply(make_object_json_serializable).unique()
        )

        logging.warning(
            f"transformed column {target} to have {num_buckets} buckets; printing the first few elements: {df[target][:5]}"
        )
    target_column_unique_values = target_candidates_and_unique_values[target]

    info: List[str] = []

    for _, row in df.iterrows():
        # If the number of choices for the target column exceeds data_args.max_target_choices,
        # take a uniform sample. Otherwise give the full list of choices for the column.
        if len(target_column_unique_values) > data_args.max_target_choices:
            target_value = make_object_json_serializable(row[target])
            target_choices = [target_value] + np.random.choice(
                [x for x in target_column_unique_values if x != target_value],
                data_args.max_target_choices - 1,
                replace=False,
            ).tolist()
        else:
            target_choices = target_column_unique_values.tolist()
        row_info = {
            "target": target,
            "target_choices": target_choices,
            "task": file,
        }
        try:
            info.append(json.dumps(row_info))
        except TypeError as te:
            logging.warning(
                f"got TypeError processing dataset {file}: {te}"
                f"row_info is {row_info}"
            )
            raise DatasetTypeError(str(te))

    df_out = df_to_records(df)
    df_out["info"] = info
    return df_out


def prepare_hf_dataset_from_formatted_df(df, as_iterable: bool):
    records = df.to_dict(orient="records")

    def _gen():
        for x in records:
            yield x

    if as_iterable:
        return datasets.IterableDataset.from_generator(_gen)

    else:
        return datasets.Dataset.from_generator(_gen)


def load_uncached_hf_dataset(
    task: str, split: str, data_args: DataArguments, as_iterable: bool
) -> Union[Dataset, IterableDataset]:
    preprocessor_config = fetch_preprocessor_config_from_data_args(data_args, task)
    if data_args.from_files:
        df = build_formatted_df_from_file(task, data_args)
    else:
        tabular_dataset = get_task_dataset(
            task, preprocessor_config=preprocessor_config
        )
        info = get_dataset_info(tabular_dataset)
        df = build_formatted_df(
            tabular_dataset._get_split_df(split),
            info,
            data_args,
        )

    return prepare_hf_dataset_from_formatted_df(df, as_iterable)


def example_map_fn(
    elem: Dict,
    data_args: DataArguments,
    serializer: RowSerializer,
    cfg: Optional[TLMConfig] = None,
) -> Dict[str, str]:
    """Extract the target from elem and serialize the other features.

    This is the main function that preprocesses individual elements
    from a key-value 'tabular' format into a text format, with the following
    fields: 'text', 'class_label_as_text'.
    """
    info = json.loads(elem["info"])
    tgt_col = info["target"]

    data = json.loads(elem["data"])
    meta = elem.pop("__metafeatures__", None)
    tgt = str(data.pop(tgt_col))

    if data_args.use_config:
        assert cfg is not None, "config is required when use_config is True."

    elif cfg is None:
        # Case: No global config. Make a config on the fly because it could be different for
        # every element of the dataset.
        cfg = TLMConfig(
            prefix=f"Predict the {tgt_col}",
            suffix=f"What is the value of {tgt_col}?",
            label_values=info["target_choices"],
        )

    if data_args.targets_handling == "map":
        tgt = cfg.map_label_value(str(tgt))
    task_context = cfg.get_task_context() if data_args.use_task_context else ""

    if data_args.use_task_context:
        assert task_context, f"expected task context but got none!"

    prefix = cfg.get_prefix() if serializer.config.use_prefix else ""
    suffix = cfg.get_suffix() if serializer.config.use_suffix else ""

    choices = cfg.get_label_values()

    features_text = serializer(
        data,
        in_context_examples=None,
        prefix_text=prefix,
        suffix_text=suffix,
        choices=choices,
        task_context_text=task_context,
        meta=meta,
    )
    preprocessed = {"class_label_as_text": tgt, "text": features_text}

    return preprocessed


def serialize_dataset_fn(
    dataset: Union[Dataset, IterableDataset],
    data_args: DataArguments,
    serializer: RowSerializer,
    cfg: Optional[TLMConfig] = None,
):
    """Take a raw HF dataset and apply serialization to it."""
    _map_fn = partial(
        example_map_fn,
        data_args=data_args,
        serializer=serializer,
        cfg=cfg,
    )
    dataset = dataset.map(_map_fn).select_columns(["text", "class_label_as_text"])
    return dataset


def load_serialized_dataset(
    task,
    data_args: DataArguments,
    serializer: RowSerializer,
    split: str,
    as_iterable: bool,
    print_one_example: bool = False,
    cfg: Optional[TLMConfig] = None,
) -> Union[Dataset, IterableDataset]:
    """Load the serialized HF dataset for a task by fetching the HF dataset and serializing the results.

    If cfg is not provided, the default TLMConfig for that dataset will be used. (For most
    "normal" use cases, where we want the canonical version of a dataset, cfg should be left as
    None).
    """

    dataset = load_uncached_hf_dataset(
        task=task, split=split, data_args=data_args, as_iterable=as_iterable
    )

    if not cfg and not data_args.from_files:
        cfg = get_tlm_config(task, override_config=data_args.task_config)

    dataset = serialize_dataset_fn(dataset, data_args, serializer, cfg)

    if print_one_example:
        print(f"printing one example from {task}/{split}: {next(iter(dataset))}")
    return dataset


def load_serialized_interleaved_dataset(
    task_names: Sequence[str],
    data_args: DataArguments,
    serializer: RowSerializer,
    as_iterable: bool,
    split="train",
    print_one_example=False,
) -> Union[Dataset, IterableDataset]:
    """Serialize and interleave the examples from task_names.

    For each task in task_names, load the dataset, serialize the examples, and interleave the
    results to produce a single IterableDataset.

    The resulting dataset has elements with the keys "text' and "class_label_as_text".
    These elements do NOT contain special tokens like the EOC/EOS tokens; those are added
    by passing the result of this function to tokenize_and_preprocess_ds_dict().
    """
    dsets = {}

    for task in task_names:
        try:
            dataset = load_serialized_dataset(
                task=task,
                data_args=data_args,
                serializer=serializer,
                split=split,
                as_iterable=as_iterable,
                print_one_example=print_one_example,
            )
            dsets[task] = dataset
        except NoTargetCandidatesError:
            logging.warning(f"skipping task {task} due to NoTargetCandidatesError.")
            continue
        except DatasetTypeError as te:
            logging.warning(f"skipping task {task} due to DatasetTypeError: {te}")

    # TODO: This is effectively a no-op for the case where task_names==1; should be if/else.
    dset = datasets.interleave_datasets(
        list(dsets.values()), stopping_strategy="all_exhausted"
    )

    return dset


def tokenize_batch(
    batch: Dict[str, List[str]],
    tokenizer,
    data_arguments: DataArguments,
) -> Dict[str, List[str]]:
    """Tokenize a dataset (in the format expected by Hugging Face dataset.map()).

    Note that this function previously also dropped too-long inputs, but this was the cause of a bug. Instead,
    the correct approach is to filter your dataset with
    ds.filter(lambda example: len(example["input_ids"]) > tokenizer.model_max_length) if you want to drop examples.
    """
    tokenized = preprocess(
        batch["input_text"],
        batch["target_text"],
        tokenizer,
    )
    if data_arguments.handle_too_long in ("drop", "warn"):
        input_ids = tokenized["input_ids"]
        too_long_idxs = [
            i
            for i in range(len(input_ids))
            if len(input_ids[i]) > tokenizer.model_max_length
        ]
        if len(too_long_idxs):
            # Case: there are inputs that exceed the max len;
            # take the action specified in data_arguments.handle_too_long.
            logging.warning(f"got {len(too_long_idxs)} inputs that are too long")
    return tokenized


def add_qa_and_eoc_tokens_to_example(ex):
    """Add special tokens (QA sep/EOC) to input/target text."""
    input = ex["text"] + QA_SEP_TOKEN
    target_text = ex["class_label_as_text"] + EOC_TOKEN
    return dict(input_text=input, target_text=target_text)


def handle_too_long(
    tokenized_ds_dict: Dict[str, Union[Dataset, IterableDataset]],
    data_arguments: DataArguments,
    max_length: int,
) -> Dict[str, Union[Dataset, IterableDataset]]:
    """Apply the correct handling of too-long inputs to each dataset in ds_dict."""
    if data_arguments.handle_too_long == "drop":
        tokenized_ds_dict = {
            split: ds.filter(lambda example: len(example["input_ids"]) <= max_length)
            for split, ds in tokenized_ds_dict.items()
        }
    return tokenized_ds_dict


def maybe_cast_to_tensor(x: Union[torch.Tensor, List, np.array]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        return torch.Tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"unknown dtype: {type(x)}")


def table_id_from_key(k: str) -> str:
    return k.split("__")[0]


def ids_and_lens_from_batch(
    batch: Dict[str, List[Union[torch.Tensor, str]]]
) -> List[List[int]]:
    """Return a nested list where the ith element is [i, len(example_i)]."""
    return [[i, len(ids)] for i, ids in enumerate(batch["input_ids"])]


def merge_batch_samples_by_key(
    batch: Dict[str, List[Union[torch.Tensor, str]]]
) -> List[List[int]]:
    """When samples are from the same source table, combine a sample with the one preceding it.

    Returns two lists of length len(batch["input_ids"]):
        ids: list where the iths element indicates the example id of the ith element in the batch. Elements from the
            same source table have the same ids. ids start from zero and count up consecutively.
        lens: list where the iths element indicates the length of the ith element in the batch.
    """
    ids_and_lens = ids_and_lens_from_batch(batch)
    for i in range(1, len(batch["__key__"])):
        if table_id_from_key(batch["__key__"][i]) == table_id_from_key(
            batch["__key__"][i - 1]
        ):
            # print(
            #     f"[DEBUG] merging samples with key {batch['__key__'][i]} and {batch['__key__'][i-1]}"
            # )
            ids_and_lens[i][0] = ids_and_lens[i - 1][0]
        else:
            # print(
            #     f"[DEBUG] NOT merging samples with key {batch['__key__'][i]} and {batch['__key__'][i - 1]}"
            # )
            # Ensure sample IDs are contiguous
            ids_and_lens[i][0] = ids_and_lens[i - 1][0] + 1
    return ids_and_lens


def generate_position_ids(ids_and_lens, max_len) -> List[int]:
    # Initialize an empty array for position indices
    position_ids = []

    # Track the current ID and initialize a position counter
    current_id = None
    current_position = 0

    for id_, length in ids_and_lens:
        # If the ID changes, reset the current position counter
        if id_ != current_id:
            current_id = id_
            current_position = 0

        # Append a range of numbers from current_position to current_position+length to the position_ids array
        position_ids.extend(
            np.arange(current_position, current_position + length).tolist()
        )
        current_position += length
        if len(position_ids) >= max_len:
            break

    return position_ids[:max_len]


def pack_samples(
    batch: Dict[str, Union[str, List[torch.Tensor]]],
    max_len: int,
    trim_extra_bos_tokens: bool = False,
    merge_samples_by_key: bool = True,
    bos_token_id: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """ "Pack a set of samples into a batch, discarding any extra data.

    The resulting dict has keys ['input_ids', 'labels', 'example_ids', 'position_ids'].
    """
    assert len(batch["input_ids"]) == len(
        batch["labels"]
    ), f"expected equal-length inputs and labels, got {len(batch['input_ids'])} and {len(batch['labels'])}"

    if trim_extra_bos_tokens and len(batch["input_ids"]) > 1:
        assert (
            bos_token_id is not None
        ), "bos_token_id is required to trim extra bos tokens."
        for i in range(1, len(batch["input_ids"])):
            if batch["input_ids"][i][0] == bos_token_id:
                batch["input_ids"][i] = batch["input_ids"][i][1:]
                batch["labels"][i] = batch["labels"][i][1:]

    # example_ids is a sequence where the integer at each sequence identifies the index of the sample
    # in the batch from which that token originated; this allows to construct an example-wise
    # attention matrix. Note that the attention matrix also needs to account for masking
    # (the attention matrix should mask tokens where labels != IGNORE_INDEX).
    # For example, it looks like [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, ...]
    if merge_samples_by_key:
        ids_and_lens = merge_batch_samples_by_key(batch)
    else:
        ids_and_lens = ids_and_lens_from_batch(batch)

    example_ids = [[i] * ids_len for i, ids_len in ids_and_lens]
    example_ids = [i for ids in example_ids for i in ids][:max_len]

    # position_ids gives the positional index of an element within its sequence.
    # For example, it looks like [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, ...]
    position_ids = generate_position_ids(ids_and_lens, max_len)

    input_ids = torch.cat([maybe_cast_to_tensor(x) for x in batch["input_ids"]], dim=0)
    input_ids = input_ids[:max_len]
    labels = torch.cat([maybe_cast_to_tensor(x) for x in batch["labels"]], dim=0)
    labels = labels[:max_len]

    return {
        "input_ids": [input_ids],
        "labels": [labels],
        "example_ids": [example_ids],
        "position_ids": [position_ids],
    }


def make_few_shot_sample(
    shots: Union[List[Tuple[torch.Tensor, torch.Tensor]], None],
    target_sample: Tuple[torch.Tensor, torch.Tensor],
    max_len: int,
    trim_extra_bos_tokens: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an (input_ids, targets) tuple from a set of shots and a target sample.

    Both the shots and the target_sample should be tokenized instances from a dataset.
    """
    if shots is None:
        shots = []

    input_ids_tensors = []
    labels_tensors = []

    for i in range(len(shots)):
        input_ids, labels = shots[i]

        if trim_extra_bos_tokens and i > 0:
            input_ids = input_ids[1:]  # drop BOS token except for the first sample
            labels = labels[1:]

        input_ids_tensors.append(input_ids)
        labels_tensors.append(torch.full_like(labels, IGNORE_INDEX))

    target_input_ids, target_labels = target_sample

    if trim_extra_bos_tokens and len(shots):
        target_input_ids = target_input_ids[1:]  # also trim the BOS token if necessary
        target_labels = target_labels[1:]

    input_ids_tensors.append(target_input_ids)
    labels_tensors.append(target_labels)

    # Concatenate and trim
    input_ids = torch.cat(input_ids_tensors, dim=0)
    input_ids = input_ids[:max_len]
    labels = torch.cat(labels_tensors, dim=0)
    labels = labels[:max_len]

    return input_ids, labels


def make_few_shot_from_labeled_batch(
    batch: Dict[str, List[torch.Tensor]],
    max_len: int,
    trim_extra_bos_tokens: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """
    Pack a set of samples (where every instance has a ground-truth label)
    into a few-shot example, where each dictionary in the returned list
    ends with a different sample in the batch as the target.
    """
    assert len(batch["input_ids"]) == len(
        batch["labels"]
    ), f"expected equal-length inputs and labels, got {len(batch['input_ids'])} and {len(batch['labels'])}"
    batch_size = len(batch["input_ids"])

    results = defaultdict(list)

    for target_sample_idx in range(batch_size):
        # For every sample, ensure the shots are in a different random order
        idxs_shuffled = random.sample(range(batch_size), batch_size)

        # Prepare the shots as a list of [(input_ids, labels) tuples.
        shots: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                maybe_cast_to_tensor(batch["input_ids"][i]),
                maybe_cast_to_tensor(batch["labels"][i]),
            )
            for i in idxs_shuffled
            if i != target_sample_idx
        ]

        # Prepare target sample with the same format as the elements of shots.
        target_sample: Tuple[torch.Tensor, torch.Tensor] = (
            maybe_cast_to_tensor(batch["input_ids"][target_sample_idx]).long(),
            maybe_cast_to_tensor(batch["labels"][target_sample_idx]).long(),
        )

        # Store in the results list
        input_ids, labels = make_few_shot_sample(
            shots,
            target_sample,
            max_len=max_len,
            trim_extra_bos_tokens=trim_extra_bos_tokens,
        )
        results["input_ids"].append(input_ids)
        results["labels"].append(labels)

    return results


def tokenize_ds(
    ds: Union[Dataset, IterableDataset],
    tokenizer,
    data_arguments: DataArguments,
):
    """Tokenize a dataset."""
    _batch_tokenize_fn = partial(
        tokenize_batch,
        tokenizer=tokenizer,
        data_arguments=data_arguments,
    )
    return ds.map(
        _batch_tokenize_fn,
        batched=True,
        batch_size=data_arguments.tokenize_fn_batch_size,
    )


def tokenize_ds_dict(
    ds_dict: Dict[str, Union[Dataset, IterableDataset]],
    tokenizer,
    data_arguments: DataArguments,
):
    """Tokenize a dataset dictionary."""

    return {
        split: tokenize_ds(ds, tokenizer=tokenizer, data_arguments=data_arguments)
        for split, ds in ds_dict.items()
    }


def tokenize_and_preprocess_ds_dict(
    ds_dict: Dict[str, Union[Dataset, IterableDataset]],
    tokenizer,
    data_arguments: DataArguments,
    is_train=True,
    max_samples: Optional[int] = None,
) -> Dict[str, Union[Dataset, IterableDataset]]:
    """Take the results of load_serialized_interleaved_dataset() and tokenize/preprocess.

    Together with load_serialized_interleaved_dataset(), this encapsulates the full dataset creation pipeline.
    """
    if max_samples is not None:
        if data_arguments.num_shots:
            max_samples = max_samples * (data_arguments.num_shots + 1)
        for split in ds_dict.keys():
            if len(ds_dict[split]) > max_samples:
                ds_dict[split] = ds_dict[split].select(range(max_samples))

    ds_dict = {
        split: ds.map(add_qa_and_eoc_tokens_to_example) for split, ds in ds_dict.items()
    }

    tokenized_ds_dict = tokenize_ds_dict(
        ds_dict, tokenizer=tokenizer, data_arguments=data_arguments
    )

    if is_train and data_arguments.pack_samples:
        tokenized_ds_dict = {
            split: ds.map(
                pack_samples,
                batched=True,
                batch_size=data_arguments.pack_samples_batch_size,
                fn_kwargs={
                    "max_len": tokenizer.model_max_length,
                    "merge_samples_by_key": data_arguments.merge_samples_by_key,
                },
            )
            for split, ds in tokenized_ds_dict.items()
        }
    elif data_arguments.num_shots:
        tokenized_ds_dict = {
            split: ds.select_columns(["input_ids", "labels"]).map(
                make_few_shot_from_labeled_batch,
                batched=True,
                batch_size=data_arguments.num_shots + 1,
                fn_kwargs={
                    "max_len": tokenizer.model_max_length,
                    "trim_extra_bos_tokens": data_arguments.trim_extra_bos_tokens,
                },
            )
            for split, ds in tokenized_ds_dict.items()
        }

    tokenized_ds_dict = handle_too_long(
        tokenized_ds_dict, data_arguments, tokenizer.model_max_length
    )
    return tokenized_ds_dict


def load_tokenize_and_serialize_tabular_dataset(
    tokenizer,
    task_names,
    data_arguments: DataArguments,
    serializer: RowSerializer,
    is_train=True,
    splits=("train", "validation", "test"),
    as_iterable: bool = True,
    max_samples: Optional[int] = None,
    print_one_example: bool = False,
) -> Dict[str, Union[Dataset, IterableDataset]]:
    """Load a tabular dataset, tokenize and preprocess it.

    Interleaves datasets from all tasks in task_names.
    """

    ds_dict = {
        split: load_serialized_interleaved_dataset(
            task_names,
            data_arguments,
            split=split,
            serializer=serializer,
            as_iterable=as_iterable,
            print_one_example=print_one_example,
        ).shuffle()
        for split in splits
    }

    tokenized_ds_dict = tokenize_and_preprocess_ds_dict(
        ds_dict=ds_dict,
        tokenizer=tokenizer,
        data_arguments=data_arguments,
        is_train=is_train,
        max_samples=max_samples,
    )

    return tokenized_ds_dict


def shuffle_dataset(
    ds: Union[Dataset, IterableDataset], shuffle_random_seed, shuffle_buffer_size=None
):
    """Helper function to shuffle a dataset, because IterableDataset and Dataset have different shuffle interfaces."""
    if isinstance(ds, IterableDataset):
        return ds.shuffle(
            seed=shuffle_random_seed,
            buffer_size=shuffle_buffer_size,
        )
    elif isinstance(ds, Dataset):
        return ds.shuffle(seed=shuffle_random_seed)
    else:
        raise ValueError(f"unsupported dataset type: {type(ds)}")


def load_and_tokenize_preserialized_hf_dataset(
    tokenizer,
    task_names,
    data_arguments: DataArguments,
    split: str,
    is_train=True,
    as_iterable: bool = True,
    max_samples: Optional[int] = None,
    shuffle=None,
    shuffle_buffer_size: Optional[int] = 10_000,
    shuffle_random_seed=42,
) -> Dict[str, Union[Dataset, IterableDataset]]:
    if shuffle is None and is_train:
        shuffle = True
    ds_dict: Dict[str, Union[Dataset, IterableDataset]] = {
        split: load_dataset(
            "parquet",
            data_files={split: task_names},
            split=split,
            streaming=as_iterable,
        )
    }
    if shuffle:
        ds_dict = {
            k: shuffle_dataset(
                ds,
                shuffle_random_seed=shuffle_random_seed,
                shuffle_buffer_size=shuffle_buffer_size,
            )
            for k, ds in ds_dict.items()
        }
    tokenized_ds_dict = tokenize_and_preprocess_ds_dict(
        ds_dict=ds_dict,
        tokenizer=tokenizer,
        data_arguments=data_arguments,
        is_train=is_train,
        max_samples=max_samples,
    )

    return tokenized_ds_dict


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def load_and_tokenize_preserialized_wds(
    tokenizer,
    urls: Sequence[str],
    data_arguments: DataArguments,
    split: str,
    is_train=True,
    as_iterable: bool = True,
    max_samples: Optional[int] = None,
    shuffle_shards: bool = True,
    shuffle_before_packing: bool = False,
    shuffle_after_packing: bool = True,
    shuffle_buffer_size: Optional[int] = 10_000,
    shuffle_random_seed=42,
    require_full_context_size: bool = True,
    shards_shuffle_buffer_size=100,
) -> Dict[str, wds.WebDataset]:
    del as_iterable
    del max_samples

    if urls[0].startswith("s3://"):
        logging.warning(f"s3 file urls detected; attempting to pipe data from s3")
        urls = [f"pipe:aws s3 cp {url} -" for url in urls]

    def _extract_json(sample) -> Dict[str, str]:
        """Fetch the {'text': ..., 'class_label_as_text': ...} for a sample."""
        key = [x for x in sample.keys() if x.endswith("json")][0]
        json_bytes = sample[key]
        return json.loads(json_bytes.decode("utf-8"))

    def _tokenize_fn(example):
        preprocessed = preprocess(
            [example["input_text"]],
            [example["target_text"]],
            tokenizer=tokenizer,
        )
        preprocessed["__key__"] = example["__key__"]
        return preprocessed

    def _flatten_values(example: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {k: v[0] for k, v in example.items()}

    def _pack_samples(x: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Take a 'batch' of samples and pack it."""
        # We have to do some weird acrobatics with the samples here to ensure we can use the exact same preprocessing
        # functions with the HF datasets and the webdatasets.
        assert len(x) == 1, f"expected len(x)==1, got len(x)=={len(x)}"
        samples = x[0]
        # Each element of samples is a dict, with the keys ['input_ids', 'labels', '__key__'].
        # Those elements contain values of a *list* where each list has a single element
        # (i.e. the input_ids, labels, or key).
        batch = {
            "input_ids": [x["input_ids"][0] for x in samples],
            "labels": [x["labels"][0] for x in samples],
            "__key__": [x["__key__"] for x in samples],
        }

        packed = pack_samples(
            batch,
            tokenizer.model_max_length,
            trim_extra_bos_tokens=data_arguments.trim_extra_bos_tokens,
            bos_token_id=tokenizer.bos_token_id,
            merge_samples_by_key=data_arguments.merge_samples_by_key,
        )
        # Returns a dict with keys ['input_ids', 'labels', 'example_ids', 'position_ids'], where we have unpacked
        # the 'HF-style' batch formatting {str: List[Tensor]} to a 'torch style' batch formatting {str: Tensor}.
        return _flatten_values(packed)

    def _make_few_shot(x: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        assert len(x) == 1, f"expected len(x)==1, got len(x)=={len(x)}"
        samples = x[0]
        # Each element of samples is a dict, with the keys ['input_ids', 'labels', '__key__'].
        # Those elements contain values of a *list* where each list has a single element
        # (i.e. the input_ids, labels, or key).
        batch = {
            "input_ids": [x["input_ids"][0] for x in samples],
            "labels": [x["labels"][0] for x in samples],
            "__key__": [x["__key__"] for x in samples],
        }
        processed = make_few_shot_from_labeled_batch(
            batch,
            tokenizer.model_max_length,
            trim_extra_bos_tokens=data_arguments.trim_extra_bos_tokens,
        )
        # Returns a dict with keys ['input_ids', 'labels', 'example_ids', 'position_ids'], where we have unpacked
        # the 'HF-style' batch formatting {str: List[Tensor]} to a 'torch style' batch formatting {str: Tensor}.
        return _flatten_values(processed)

    def _filter_fn(example) -> bool:
        """Return True if example length is less than ir equal to tokenizer.model_max_length."""
        if is_train and require_full_context_size:
            # Require samples to be exactly length tokenizer.model_max_length
            length_is_ok = len(example["input_ids"]) == tokenizer.model_max_length
        elif is_train:
            # Require samples to fit in context window
            length_is_ok = len(example["input_ids"]) <= tokenizer.model_max_length
        else:
            # For non-training cases, we consider padding tokens, and require samples
            # to fit in context window.
            length_is_ok = (
                example["input_ids"].ne(tokenizer.pad_token_id).sum().item()
                + example["labels"].ne(-100).sum().item()
                <= tokenizer.model_max_length
            )
        if not length_is_ok:
            logging.warning(f"dropping sample with length {len(example['input_ids'])}")
        return length_is_ok

    pipeline = [
        wds.SimpleShardList(urls, seed=shuffle_random_seed),
    ]

    if shuffle_shards:
        # at this point we have an iterator over all the shards
        pipeline.append(
            wds.shuffle(shards_shuffle_buffer_size),
        )

    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ]
    )

    if shuffle_before_packing:
        # This will pack random/unrelated samples together if activated
        pipeline.append(wds.shuffle(shuffle_buffer_size))

    pipeline.extend(
        [
            wds.map(_extract_json),
            wds.map(add_qa_and_eoc_tokens_to_example),
            wds.map(_tokenize_fn),
        ]
    )

    if data_arguments.pack_samples:
        pipeline.extend(
            [
                # Elements must be lists for batching
                wds.map(lambda x: [x]),
                wds.batched(data_arguments.pack_samples_batch_size),
                wds.map(_pack_samples),
            ]
        )
    elif data_arguments.num_shots > 1:
        pipeline.extend(
            [
                # Elements must be lists for batching
                wds.map(lambda x: [x]),
                wds.batched(data_arguments.num_shots),
                wds.map(_make_few_shot),
            ]
        )
    else:
        pipeline.append(wds.map(_flatten_values))

    if data_arguments.handle_too_long == "drop":
        pipeline.append(wds.select(_filter_fn))

    if shuffle_after_packing:
        pipeline.append(wds.shuffle(shuffle_buffer_size))

    dataset = wds.DataPipeline(*pipeline)

    return {split: dataset}


def load_and_tokenize_preserialized_dataset(
    tokenizer,
    task_names,
    data_arguments: DataArguments,
    split: str,
    is_train=True,
    as_iterable: bool = True,
    max_samples: Optional[int] = None,
    shuffle=None,
    shuffle_buffer_size: Optional[int] = 10_000,
    shuffle_random_seed=42,
) -> Dict[str, Union[Dataset, IterableDataset]]:
    """Load a preserialized tabular dataset, tokenize and preprocess it.

    Interleaves datasets from all tasks in task_names.
    """
    if task_names[0].endswith(".wds") or task_names[0].endswith(".tar"):
        return load_and_tokenize_preserialized_wds(
            tokenizer=tokenizer,
            urls=task_names,
            data_arguments=data_arguments,
            split=split,
            is_train=is_train,
            as_iterable=as_iterable,
            max_samples=max_samples,
            shuffle_before_packing=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_random_seed=shuffle_random_seed,
        )
    else:
        return load_and_tokenize_preserialized_hf_dataset(
            tokenizer=tokenizer,
            task_names=task_names,
            data_arguments=data_arguments,
            split=split,
            is_train=is_train,
            as_iterable=as_iterable,
            max_samples=max_samples,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_random_seed=shuffle_random_seed,
        )


def prepare_tokenized_dataset(
    data_arguments: DataArguments,
    train_config: TrainConfig,
    serializer,
    accelerator,
    tokenizer,
    task_names,
    shuffle: bool,
    split="train",
):
    if not data_arguments.use_preserialized:
        train_ds_tokenized = load_tokenize_and_serialize_tabular_dataset(
            tokenizer=tokenizer,
            task_names=task_names,
            data_arguments=data_arguments,
            serializer=serializer,
            print_one_example=accelerator.is_main_process,
        )
    else:
        train_ds_tokenized = load_and_tokenize_preserialized_dataset(
            tokenizer=tokenizer,
            task_names=task_names,
            data_arguments=data_arguments,
            split=split,
            shuffle=shuffle,
            shuffle_buffer_size=train_config.shuffle_buffer_size,
            shuffle_random_seed=train_config.shuffle_random_seed,
        )
    return train_ds_tokenized
