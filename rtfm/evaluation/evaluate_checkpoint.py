"""
Evaluate a language model for tabular data prediction.

Models should be in Hugging Face format.
"""
import logging
import os
from typing import Dict, Optional

import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from rtfm.arguments import DataArguments
from rtfm.configs import TrainConfig, SerializerConfig
from rtfm.evaluation.evaluation_utils import (
    prepare_eval_kwargs,
    prepare_eval_datasets,
)
from rtfm.evaluation.evaluators import build_evaluators, ClosedVocabularyEvaluator
from rtfm.hf_utils import fetch_auth_token
from rtfm.serialization.serializers import get_serializer
from rtfm.task_config import get_tlm_config
from rtfm.utils import get_task_names_list, initialize_dir

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from datasets import disable_caching

logging.warning("disabling caching for hf datasets!")
disable_caching()

transformers.logging.set_verbosity_info()


def main(
    data_arguments: DataArguments,
    train_config: TrainConfig,
    serializer_config: SerializerConfig,
    outfile: str,
    split: str,
    eval_task_names: Optional[str] = None,
    eval_task_file: Optional[str] = None,
    overwrite: bool = False,
):
    if os.path.exists(outfile) and not overwrite:
        logging.warning(f"file {outfile} already exists; skipping evaluation.")
        return
    assert not (
        eval_task_names and eval_task_file
    ), "specify either eval_task_names or eval_task_file, not both."

    assert (
        not data_arguments.pack_samples
    ), "are you sure you want to use packing for evals?!"

    assert outfile.endswith(".csv"), "output file must end with .csv"
    initialize_dir(os.path.dirname(outfile))

    eval_task_names = get_task_names_list(
        eval_task_names if eval_task_names else eval_task_file
    )

    assert len(eval_task_names) == 1 or (not data_arguments.tag), (
        f"it is recommended to use this script for only a single task to ensure"
        f"tags and context shift are applied correctly, when tags are used."
        f"Either do not use a tag, or use a single eval task name."
    )

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        return_dict=True,
        load_in_8bit=train_config.quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name, token=fetch_auth_token()
    )
    serializer = get_serializer(serializer_config)

    assert not train_config.resume, (
        f"using TrainConfig.resume is deprecated; instead export the model"
        f"and tokenizer to hugging face format and provide the"
        f"path to the directory containing the saved model/tokenizer"
        f" as the --model_name flag."
    )

    splits_to_keep = ("train", "validation", "test") if not eval_task_file else None
    print(f"splits_to_keep is {splits_to_keep}")
    eval_dataset_kwargs = prepare_eval_kwargs(
        tokenizer=tokenizer,
        eval_serializer=serializer,
        accelerator=None,
        data_arguments=data_arguments,
        train_config=train_config,
        splits_to_keep=splits_to_keep,
    )
    eval_datasets_tokenized = prepare_eval_datasets(
        eval_task_names=eval_task_names,
        exclude_task_names=None,
        data_arguments=data_arguments,
        splits_to_keep=splits_to_keep,
        **eval_dataset_kwargs,
    )

    evaluators = build_evaluators(train_config)

    output_metrics: Dict[str, float] = {}
    for eval_task_name, eval_task_dataset in eval_datasets_tokenized.items():
        prefix = f"{split}/{eval_task_name}"

        for evaluator in evaluators:
            if data_arguments.use_config and isinstance(
                evaluator, ClosedVocabularyEvaluator
            ):
                eval_task_config = get_tlm_config(
                    eval_task_name.replace("_holdout", "")
                )

                label_values = eval_task_config.get_label_values()
            else:
                label_values = None

            metrics = evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                train_config=train_config,
                dataset=eval_task_dataset,
                wandb_logging_prefix=prefix,
                step=None,
                labels=label_values,
            )
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            output_metrics.update(metrics)
            # TODO: log metrics to wandb.
            print(metrics)

    df = (
        pd.DataFrame.from_dict({k: [v] for k, v in output_metrics.items()})
        .T.reset_index()
        .rename(columns={"index": "metric", 0: "value"})
    )
    df["model_arguments.model_name_or_path"] = train_config.model_name
    df["split"] = split
    df.to_csv(outfile, index=False)
    print(df)
    print(f"metrics written to {outfile}")

    return


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (DataArguments, TrainConfig, SerializerConfig)
    )

    parser.add_argument(
        "--eval-task-names",
        required=False,
        type=str,
        default=None,
        help="Comma-delimited list of names of task(s) to use for evaluation.",
    )

    parser.add_argument("--eval-task-file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        default="test",
        help="metrics_prefix of data to use.",
    )

    parser.add_argument(
        "--outfile",
        required=True,
        help="where to write the metrics files for each task.",
    )

    (
        data_args,
        train_config,
        serializer_config,
        other_args,
    ) = parser.parse_args_into_dataclasses()

    main(
        data_arguments=data_args,
        train_config=train_config,
        serializer_config=serializer_config,
        **vars(other_args),
    )
