from typing import List, Dict, Any, Sequence
from typing import Union, Optional

import datasets
import transformers
from accelerate import Accelerator
from datasets import Dataset
from tqdm import tqdm

from rtfm.arguments import DataArguments
from rtfm.configs import TrainConfig
from rtfm.data import (
    load_tokenize_and_serialize_tabular_dataset,
    load_and_tokenize_preserialized_dataset,
)
from rtfm.generation_utils import parse_generated_text
from rtfm.utils import get_task_names_list


def get_preds(
    tokenized_ds: Dataset,
    max_eval_samples: int,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.AutoModelForCausalLM,
    max_new_tokens: int = 6,
) -> List[Dict]:
    preds = []
    iterator = iter(tokenized_ds)

    for _ in tqdm(range(max_eval_samples), total=max_eval_samples):
        try:
            x = next(iterator)
        except StopIteration:
            break

        # Instead of using the collator, which risks issues with the masking
        # of the inputs, we use only the input_text of x -- this is raw text
        # + separator -- to ensure that there is no label leakage.
        input_text = x["input_text"]
        input_ids = tokenizer(
            input_text, return_attention_mask=False, return_tensors="pt"
        )["input_ids"]
        input_ids = input_ids.cuda()
        outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        outputs_parsed = [parse_generated_text(x)[0] for x in outputs_decoded]
        preds.append({**x, "outputs_parsed": outputs_parsed})

    return preds


def prepare_eval_kwargs(
    tokenizer,
    eval_serializer,
    accelerator: Union[Accelerator, None],
    data_arguments: DataArguments,
    train_config: TrainConfig,
    splits_to_keep: Sequence[str] = ("test",),
) -> Dict[str, Any]:
    eval_dataset_kwargs = {
        "tokenizer": tokenizer,
        "is_train": False,
        "as_iterable": False,
        "max_samples": train_config.eval_max_samples,
    }
    if not data_arguments.use_preserialized:
        eval_dataset_kwargs.update(
            {
                "serializer": eval_serializer,
                "splits": splits_to_keep,
                "print_one_example": accelerator.is_main_process
                if accelerator is not None
                else True,
            }
        )
    return eval_dataset_kwargs


def prepare_eval_datasets(
    eval_task_names: Union[str, None],
    exclude_task_names: Union[Sequence[str], None],
    data_arguments: DataArguments,
    dict_key_or_suffix: str = "holdout",
    splits_to_keep: Optional[Sequence[str]] = ("test",),
    **kwargs,
) -> Dict:
    if exclude_task_names is None:
        exclude_task_names = []

    eval_datasets_tokenized = {}
    if eval_task_names and not data_arguments.use_preserialized:
        # For data that is not preserialized we prepare a separate dataset with each task's data.
        for task_name in eval_task_names:
            if task_name not in exclude_task_names:
                k = task_name + "_" + dict_key_or_suffix
                tokenized_and_serialized = load_tokenize_and_serialize_tabular_dataset(
                    task_names=[task_name], data_arguments=data_arguments, **kwargs
                )
                ds = datasets.concatenate_datasets(
                    [tokenized_and_serialized[split] for split in splits_to_keep]
                )
                eval_datasets_tokenized[k] = ds

    elif eval_task_names and data_arguments.use_preserialized:
        # For preserialized data we have a single eval task with all shards.
        assert not splits_to_keep, (
            f"cannot specify splits_to_keep with preserialized datasets; "
            f"got splits_to_keep={splits_to_keep}"
        )
        tokenized_and_serialized = load_and_tokenize_preserialized_dataset(
            task_names=eval_task_names,
            split="test",
            shuffle=True,
            data_arguments=data_arguments,
            **kwargs,
        )
        ds = tokenized_and_serialized["test"]
        # ds = datasets.concatenate_datasets(
        #     [tokenized_and_serialized[split] for split in splits_to_keep]
        # )
        eval_datasets_tokenized[dict_key_or_suffix] = ds

    else:
        # In this case, there are no eval datasets. We only use the train_eval datasets.
        eval_datasets_tokenized = {}
    return eval_datasets_tokenized


def prepare_train_eval_datasets(
    train_task_names, train_eval_task_file, data_arguments: DataArguments, **kwargs
):
    # Train_eval datasets; these are the test splits of the training tasks.
    if not data_arguments.use_preserialized:
        train_eval_datasets_tokenized = {
            task_name: load_tokenize_and_serialize_tabular_dataset(
                task_names=[task_name], data_arguments=data_arguments, **kwargs
            )["test"]
            for task_name in train_task_names
        }
    elif data_arguments.use_preserialized and train_eval_task_file:
        train_eval_task_names = get_task_names_list(train_eval_task_file)
        train_eval_datasets_tokenized = {
            "train_eval": load_and_tokenize_preserialized_dataset(
                task_names=train_eval_task_names,
                split="test",
                data_arguments=data_arguments,
                shuffle=True,
                **kwargs,
            )["test"]
        }
    else:
        raise ValueError
    return train_eval_datasets_tokenized
