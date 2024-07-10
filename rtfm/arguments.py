import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Literal

from rtfm.configs import TrainConfig

# Mapping of command-line-friendly feature transform strings to
# tableshift-friendly strings.

CAT_FEATURE_TRANSFORM_MAPPING = {
    "map": "map_values",
    "none": "passthrough",
    "le": "label_encode",
    "normalize": "passthrough",  # when we normalize numeric features, passthrough categorical features.
}
NUM_FEATURE_TRANSFORM_MAPPING = {
    "map": "map_values",
    "none": "passthrough",
    "normalize": "normalize",
    # 'label encoding' of numeric features is a no-op.
    "le": "passthrough",
}


@dataclass
class LoggingArguments:
    log_to_wandb: bool = False
    wandb_project: str = "rtfm"


@dataclass
class ModelArguments:
    pass


@dataclass
class DataArguments:
    use_position_ids: bool = field(
        default=True,
        metadata={"help": "Whether to use position_ids during training."},
    )
    pack_samples: bool = field(
        default=True, metadata={"help": "Whether to pack batches of samples."}
    )
    pack_samples_batch_size: int = field(
        default=150, metadata={"help": "Number of samples to pack into a batch."}
    )
    merge_samples_by_key: bool = field(
        default=True,
        metadata={
            "help": "Whether to merge samples by the key in the webdataset when packing. "
            "This has the effect of training the model on all 'shots' from a given table,"
            "by allowing it to attend to tokens across samples."
            "If False, every row from a table is a separate example and there is no attending to other rows, even if they are from the same table."
            "If sample packing is set to False, this argument has no effect."
        },
    )
    labels_require_nonunique: bool = field(
        default=True,
        metadata={
            "help": "If True, candidate label columns are excluded if they are unique for every element."
            "This excludes fields such as UIDs or other identifiers that do not define groups in the data."
        },
    )
    labels_min_unique_values: int = field(
        default=2,
        metadata={
            "help": "Minimum number of unique values required for a candidate label column."
            "Columns with fewer than this number of unique values will not be prediction targets.."
        },
    )
    labels_drop_numeric: bool = field(
        default=False,
        metadata={
            "help": "If True, candidate label columns are excluded if all values are numeric."
        },
    )
    labels_p_numeric: float = field(
        default=0.1,
        metadata={
            "help": "Probability of selecting a numeric column, when both numeric and categorical columns are present."
            "This trades off the number of numeric columns vs. the number of nonnumeric columns in the data."
        },
    )
    labels_drop_dates: bool = field(
        default=True,
        metadata={
            "help": "If True, candidate label columns are excluded if they contain a pandas date dtype."
            "Note that string values are NOT parsed; instead we rely strictly on the data types as parsed from Arrow"
            "(this means that some dates might evade this filtering strategy without further processing)."
        },
    )
    cache_dir: str = field(
        default="tmp", metadata={"help": "Cache directory for data."}
    )
    use_preserialized: bool = field(
        default=False, metadata={"help": "if true, use a preserialized dataset."}
    )
    tag: str = field(
        default=None,
        metadata={
            "help": "if specified, this is appended to the uid of the dataset (preceded by '__')."
            "This is useful when training on data with e.g. a dataset with feature shift applied."
        },
    )
    handle_too_long: str = field(
        default="drop",
        metadata={
            "choices": ("drop", "warn"),
            "help": "How to handle inputs that are too long for the model."
            "If 'drop', inputs will be dropped and a WARNING statement will be logged for each occurrence."
            "If 'warn', inputs will not be dropped, and a WARNING statement will be logged for each occurrence.",
        },
    )
    shuffle_table_features: bool = field(
        default=False,
        metadata={
            "help": "If True, randomly shuffle the order of features for each table. "
            "(The order will still be the same for all isntances from that table unless "
            "shuffle_instance_features is set to True). Note that this willl only affect "
            "datasets that are NOT preserialized."
        },
    )
    feature_value_handling: str = field(
        default="map",
        metadata={
            "choices": list(CAT_FEATURE_TRANSFORM_MAPPING.keys()),
            "help": "how/whether to preprocess tabular data values "
            "when there is a value mapping for a feature",
        },
    )
    feature_name_handling: str = field(
        default="map",
        metadata={
            "choices": ("none", "map"),
            "help": "how/whether to preprocess tabular data names "
            "when there is an extended name for a feature",
        },
    )
    targets_handling: str = field(
        default="map",
        metadata={"choices": ("none", "map"), "help": "handling of labels."},
    )
    task_config: str = field(
        default=None,
        metadata={
            "help": "prefix of a task config in task_config directory (the part of "
            "the filename before .yaml) to use, if not using the default task config."
            "Setting the task_config allows you to use the *same* dataset "
            "with *different* configs (for example, different task contexts)."
        },
    )
    num_shots: int = field(
        default=0,
        metadata={
            "help": "Number of shots to use. If set to zero, few-shot is not used."
        },
    )
    trim_extra_bos_tokens: bool = field(
        default=True,
        metadata={
            "help": "Whether to remove <bos> tokens on 'shots' following the first shot."
            "This affects both 'packed' training samples, and few-shot eval data."
            "If True, there will be only a single <bos> token per sequence, at index zero."
        },
    )
    use_task_context: bool = field(
        default=False,
        metadata={
            "help": "whether to use task task_context. "
            "Will raise a ValueError if any dataset does not contain task task_context."
        },
    )
    use_config: bool = field(
        default=True,
        metadata={
            "help": "Whether to use a predefined TLMConfig and apply it to "
            "every example in the dataset."
            "For benchmark tasks this should be set to True."
            "For 'unsupervised' tabular training, this should be False"
        },
    )
    from_files: bool = field(
        default=False,
        metadata={
            "help": "Set to True if you expect a TaskConfig to exist for this task."
            "Set to False otherwise (i.e. if loading a task from raw files)."
        },
    )
    max_target_choices: int = field(
        default=8,
        metadata={
            "help": "Only used when from_files is True. This defines the"
            "maximum number of target classes to be included in the serialized example."
            "Target labels are sampled uniformly at random."
        },
    )
    max_target_len_chars: int = field(
        default=256,
        metadata={
            "help": "Only used when from_files is True. This defines the"
            "maximum number of characters allows in a target column. If any values in the"
            "column have more than this number of characters, it cannot be used as a target."
        },
    )

    use_metafeatures: bool = field(
        default=False,
        metadata={"help": "Whether to add the quantile transform to data."},
    )
    metafeatures_max_precision: int = field(
        default=2, metadata={"help": "number of decimal places to use for metafeatures"}
    )
    tokenize_fn_batch_size: int = field(
        default=1000,
        metadata={
            "help": "Batch size to use when tokenizing. This increases parallelism when tokenizing and can help avoid CPU-bound input pipelines."
            "2000 is also the HF default value."
        },
    )
    dropna: Literal["rows", "columns", None] = field(
        default=None,
        metadata={
            "help": "Value passed to Tableshift PreprocessorConfig."
            "See tableshift.core.features.PreprocessorConfig for details."
        },
    )

    def set_tag(self, tag):
        assert self.tag is None, f"tag is already set to value {self.tag}"
        self.tag = tag


def write_args_to_file(args: List[str], dir: str):
    """Write the arguments to a file."""
    if not os.path.exists(dir):
        os.makedirs(dir)
    fp = os.path.join(dir, "args.txt")
    if os.path.exists(fp):
        logging.info(f"args file already exists at {fp}; overwriting it.")
        try:
            os.remove(fp)
        except FileNotFoundError:
            # when running in parallel, multiple processes might try to remove
            # the file; this handles that potential race condition.
            pass
    logging.info(f"writing args to {fp}")
    with open(fp, "w") as f:
        for arg in args:
            if arg.startswith("-"):
                f.write(arg + " ")
            else:
                f.write(arg + " \\" + "\n")


def make_uid_from_args(
    data_args: DataArguments = None,
    model_args: ModelArguments = None,
    train_config: TrainConfig = None,
    add_timestamp: bool = False,
) -> str:
    """Make a unique task identifier from the arguments.

    This function tries to encapsulate the most important parameters that might
    differentiate an task, but should be updated as needed as the main axes of
    experimental variation change.

    Note that if add_timestamp=False, the UID will be the *same*
    for datasets with identical data_arguments, model_arguments,
    and training_arguments! This allows to easily recover datasets/model checkpoints with
    the same hyperparameters, but can have unintended consequences.
    """
    uid = ""
    if model_args:
        model_feats = {
            "model": os.path.basename(model_args.model_name_or_path),
            "ser": model_args.serializer_cls,
        }
        uid += "_".join(f"{k}_{v}" for k, v in model_feats.items())

    if data_args:
        data_feats = {
            "features": data_args.feature_value_handling,
            "names": data_args.feature_name_handling,
            "targets": data_args.targets_handling,
        }

        uid += "_".join(f"{k}_{v}" for k, v in data_feats.items())

    if train_config:
        uid += train_config.run_name
    if add_timestamp:
        uid += "_" + str(int(time.time()))
    if data_args.tag:
        uid = "__".join((uid, data_args.tag))
    return uid
