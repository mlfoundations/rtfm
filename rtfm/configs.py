import time
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Sequence, Any

from llama_recipes.configs.training import train_config
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class TrainConfig(train_config):
    max_steps: int = 16
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.0
    save_steps: int = None
    eval_steps: int = 1_000
    eval_max_batches: int = 100
    resume: Optional[str] = None
    save_total_limit: int = 1
    freeze_input_embeddings: bool = False
    serializer_cls: str = "BasicSerializerV2"
    # Whether to upload a CSV containing predictions to wandb.
    eval_upload_predictions: Literal["no", "on_eval", "on_finish"] = "on_eval"
    shuffle_buffer_size: int = 10_000
    shuffle_random_seed: int = 42
    eval_open_vocabulary: bool = True
    eval_closed_vocabulary: bool = False
    per_device_eval_batch_size: int = 1
    run_name: str = field(default_factory=lambda: str(int(time.time())))
    eval_max_samples: Optional[int] = 1024
    report_to: Sequence[Any] = field(default_factory=lambda: tuple())


@dataclass
class TokenizerConfig:
    """Configuration class for tokenization and serialization."""

    add_serializer_tokens: bool = True
    serializer_tokens_embed_fn: Literal["smart", "vipi", "hf"] = "smart"
    use_fast_tokenizer: bool = True


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
