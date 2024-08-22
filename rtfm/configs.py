import os.path
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Sequence, Any, Callable

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import pandas as pd


@dataclass
class TrainConfig:
    max_steps: int = 16
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.0
    save_steps: int = None
    eval_steps: int = 1_000
    eval_max_batches: int = 100
    resume: Optional[str] = None
    save_total_limit: int = 1
    freeze_input_embeddings: bool = False
    # Whether to upload a CSV containing predictions to wandb.
    eval_upload_predictions: Literal["no", "on_eval", "on_finish"] = "on_eval"
    shuffle_buffer_size: int = 10_000
    shuffle_random_seed: int = 42
    eval_open_vocabulary: bool = True
    eval_closed_vocabulary: bool = False
    eval_max_samples: Optional[int] = 1024
    report_to: Sequence[Any] = field(default_factory=lambda: tuple())
    # torch.compile args
    torch_compile: bool = False
    torch_compile_fullgraph: bool = True  # set to False if graph is not static.
    # Below are params that originally were part of
    # llama_recipes.configs.training.train_config class.
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "packing"  # alternative: padding
    context_length: int = 8192
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    peft_method: str = "None"  # None , llama_adapter, prefix, lora
    use_peft: bool = False
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    save_model: bool = False
    save_checkpoint_root_dir: str = "checkpoints"  # will be used if using FSDP
    run_name: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = True  # Enable wandb for experient tracking
    save_metrics: bool = (
        False  # saves training metrics to a json file for later plotting
    )

    @property
    def output_dir(self) -> str:
        return os.path.join(self.save_checkpoint_root_dir, self.run_name)

    def make_save_folder_name(self, step: Optional[int] = None) -> str:
        """Encapsulates the llama-recipes logic for building a folder name.

        Instead of duplicating this code in many places, we implement it in a single reusable function here.

        If step is not provided, this function returns the parent checkpoint directory for the TrainConfig.

        If step is provided, this function returns the directory for a specific step (a subdirectory inside the parent directory).
        """
        base_dir = (
            self.save_checkpoint_root_dir
            + "/"
            + self.run_name
            + "-"
            + self.model_name.split("/")[-1]
        )
        if not step:
            return base_dir
        else:
            return base_dir + "/" + f"step-{step}"


@dataclass
class TokenizerConfig:
    """Configuration class for tokenization and serialization."""

    # "whether to add special tokens for the serializer to the vocabulary. "
    # "If False, the tokens (i.e. <VALUE_START> for StructuredSerializer) are added"
    # " to the example text, but are not explicitly added as special tokens "
    # "to the tokenizer vocabulary."
    add_serializer_tokens: bool = True
    # "Embedding initialization method to use for serializer special tokens."
    # "Only used if add_serializer_tokens=True (ignored otherwise)"
    serializer_tokens_embed_fn: Literal["smart", "vipi", "hf"] = "smart"
    use_fast_tokenizer: bool = True


@dataclass
class TargetSelectorConfig:
    """Configuration class for target selection-related parameters.

    Target selection refers to the process whereby a prediction target column
    is selected from a DataFrame with no prespecified prediction target.
    """

    target_selector_cls: Literal["T4TargetSelector", "ModelBasedTargetSelector"]

    # Configuration options for ModelBasedTargetSelector
    model_path: str = os.path.join(
        os.path.dirname(__file__),
        "models",
        "xgb_target_quality_scorer_c56e00b3-e1df-4d36-a348-7b7006deba3b.json",
    )
    selection_method: Literal["max", "topk", "temperature"] = "max"
    k: int = 3
    temperature: float = 1.0

    # Parameters from the old DataArguments class implementation
    labels_require_nonunique: bool = True
    #     metadata={
    #         "help": "If True, candidate label columns are excluded if they are unique for every element."
    #                 "This excludes fields such as UIDs or other identifiers that do not define groups in the data."
    #     },
    # )
    labels_min_unique_values: int = 2
    #     metadata={
    #         "help": "Minimum number of unique values required for a candidate label column."
    #                 "Columns with fewer than this number of unique values will not be prediction targets.."
    #     },
    # )
    labels_drop_numeric: bool = False
    #     metadata={
    #         "help": "If True, candidate label columns are excluded if all values are numeric."
    #     },
    # )
    labels_p_numeric: float = 0.1
    #     metadata={
    #         "help": "Probability of selecting a numeric column, when both numeric and categorical columns are present."
    #                 "This trades off the number of numeric columns vs. the number of nonnumeric columns in the data."
    #     },
    # )
    labels_drop_dates: bool = True
    #     metadata={
    #         "help": "If True, candidate label columns are excluded if they contain a pandas date dtype."
    #                 "Note that string values are NOT parsed; instead we rely strictly on the data types as parsed from Arrow"
    #                 "(this means that some dates might evade this filtering strategy without further processing)."
    #     },
    # )
    max_target_choices: int = 8
    # field(
    #     default=8,
    #     metadata={
    #         "help": "Only used when from_files is True. This defines the"
    #         "maximum number of target classes to be included in the serialized example."
    #         "Target labels are sampled uniformly at random."
    #     },
    # )
    max_target_len_chars: int = 256
    # field(
    #     default=256,
    #     metadata={
    #         "help": "Only used when from_files is True. This defines the"
    #         "maximum number of characters allows in a target column. If any values in the"
    #         "column have more than this number of characters, it cannot be used as a target."
    #     },
    # )


@dataclass
class SerializerConfig:
    """Configuration class for serializer."""

    serializer_cls: str = "BasicSerializerV2"
    shuffle_instance_features: bool = False
    #     default=False,
    #     metadata={
    #         "help": "If true, randomly shuffle the order of features for each instance."
    #     },
    # )
    feature_dropout: float = 0.0
    #     default=0.0,
    #     metadata={
    #         "help": "Proportion of features in each example to randomly drop out during training."
    #     },
    # )
    use_prefix: bool = True
    #     default=True,
    #     metadata={
    #         "help": "whether to use a prefix for examples. The prefix lists "
    #                 "valid choices, and describes the prediction task."
    #     },
    # )
    use_suffix: bool = True
    #     default=True,
    #     metadata={
    #         "help": "Whether to use a suffix for examples. "
    #                 "The suffix phrases the prediction tasks as a question, "
    #                 "and lists valid choices."
    #     },
    # )
    use_choices: bool = True
    #     default=True,
    #     metadata={"help": "Whether to list the class choices in the prompt."},
    # )
    choices_position: Literal["front", "back", "both"] = "both"
    max_precision: Optional[int] = None


@dataclass
class LoraConfig:
    lora_r: int = 32
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class FsdpConfig:
    """Fixed some issues with llama-recipes.configs.fsdp.fsdp_config."""

    # HYBRID_SHARD "Full Shard within a node DDP cross Nodes",
    # SHARD_GRAD_OP "Shard only Gradients and Optimizer States",
    # NO_SHARD "Similar to DDP".
    sharding_strategy: Literal[
        "FULL_SHARD",
        "HYBRID_SHARD",
        "SHARD_GRAD_OP",
        "NO_SHARD",
        ShardingStrategy.FULL_SHARD,
        ShardingStrategy.HYBRID_SHARD,
        ShardingStrategy.SHARD_GRAD_OP,
        ShardingStrategy.NO_SHARD,
    ] = "FULL_SHARD"
    hsdp: bool = False  # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    sharding_group_size: int = 0  # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int = 0  # requires hsdp to be set. This specifies the replica group size, which is world_size/sharding_group_size.
    checkpoint_type: StateDictType = (
        StateDictType.FULL_STATE_DICT
    )  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"

    def __post_init__(self):
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = eval("ShardingStrategy." + self.sharding_strategy)
