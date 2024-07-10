import os
from typing import Optional, Tuple

import torch
from accelerate.utils import is_xpu_available
from llama_recipes.utils import (
    hsdp_device_mesh as make_hsdp_device_mesh,
    freeze_transformer_layers,
    get_policies,
    fsdp_auto_wrap_policy,
    setup,
)
from llama_recipes.utils.train_utils import (
    setup_environ_flags,
    clear_gpu_cache,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from transformers import LlamaForCausalLM, LlamaConfig, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from rtfm.configs import TrainConfig, FsdpConfig


def fsdp_wrap_model(
    model, train_config: TrainConfig, fsdp_config: FsdpConfig, rank: int
):
    if (
        fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh = make_hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")

    else:
        hsdp_device_mesh = None

    if not train_config.use_peft and train_config.freeze_layers:
        freeze_transformer_layers(model, train_config.num_freeze_layers)

    mixed_precision_policy, wrapping_policy = get_policies(train_config, rank)
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

    device_id = 0
    if is_xpu_available():
        device_id = torch.xpu.current_device()
    elif torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrapping_policy
        if train_config.use_peft
        else wrapping_policy,
        cpu_offload=CPUOffload(offload_params=True)
        if fsdp_config.fsdp_cpu_offload
        else None,
        mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_mesh=hsdp_device_mesh,
        device_id=device_id,
        limit_all_gathers=True,
        sync_module_states=train_config.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        )
        if train_config.low_cpu_fsdp and rank != 0
        else None,
        use_orig_params=train_config.torch_compile,
    )
    return model


def safe_setup():
    try:
        setup()
    except ValueError as ve:
        if "trying to initialize the default process group twice" in str(ve):
            return


def dist_setup(train_config) -> Tuple[int, int]:
    safe_setup()

    rank = None
    local_rank = None

    if torch.distributed.is_initialized():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    return rank, local_rank


def load_model(
    train_config: TrainConfig, fsdp_config: FsdpConfig, rank: Optional[int] = None
):
    # Load the pre-trained model and setup its configuration
    # Load the configuration
    config = AutoConfig.from_pretrained(train_config.model_name)
    if fsdp_config.pure_bf16:
        # Set the torch_dtype to bfloat16 which matches TabuLa train/eval setup
        config.torch_dtype = "bfloat16"

    use_cache = False if train_config.enable_fsdp else None

    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        assert rank is not None, "rank must be defined when using FSDP."

        # for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        # this avoids cpu oom when loading large models like llama 70B, in which case
        # model alone would consume 2+TB cpu mem (70 * 4 * 8).

        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                config=config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            config=config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )
    return model
