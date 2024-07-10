"""
Fine-tune a language model for tabular data prediction.
"""

import dataclasses
import os
import random
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available
from llama_recipes.configs import wandb_config as WandbConfig
from llama_recipes.policies import apply_fsdp_checkpointing
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
)
from llama_recipes.utils.train_utils import print_model_size
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    HfArgumentParser,
)

from rtfm.arguments import DataArguments
from rtfm.configs import TrainConfig, FsdpConfig, LoraConfig, TokenizerConfig
from rtfm.data import (
    prepare_tokenized_dataset,
    DataCollatorForSupervisedDataset,
)
from rtfm.distributed import fsdp_wrap_model, load_model, dist_setup
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import sanity_check_tokenizer, prepare_tokenizer
from rtfm.train_utils import (
    OPTIMIZER_STATE_PT,
    SCHEDULER_STATE_PT,
)
from rtfm.train_utils import train, load_model_from_checkpoint
from rtfm.utils import get_task_names_list, get_latest_checkpoint


def setup_wandb(config_dict: Dict[str, Any], **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )

    wandb_config = WandbConfig()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict, config=config_dict)
    return run


def main(
    train_config: TrainConfig,
    fsdp_config: FsdpConfig,
    lora_config: LoraConfig,
    data_arguments: DataArguments,
    tokenizer_config: TokenizerConfig,
    train_task_file: str,
    eval_task_file: Optional[str] = None,
):
    data_arguments_defaults = {
        "use_preserialized": True,
        "targets_handling": "none",
        "use_task_context": False,
        "from_files": True,
        "use_config": False,
        "pack_samples": True,
    }

    # get_policies requires FsdpConfig to have mixed_precision and use_fp16 attributes, but
    # we use the ones from TrainConfig to avoid duplicate/conflicting flags in the HFArgumentParser.
    setattr(fsdp_config, "mixed_precision", train_config.mixed_precision)
    setattr(fsdp_config, "use_fp16", train_config.use_fp16)

    for k, v in data_arguments_defaults.items():
        setattr(data_arguments, k, v)

    train_task_names = get_task_names_list(train_task_file)

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        rank, local_rank = dist_setup(train_config)

    wandb_run = None

    if train_config.use_wandb:
        config_dict = {
            **data_arguments.__dict__,
            **dataclasses.asdict(train_config),
            **dataclasses.asdict(fsdp_config),
            **dataclasses.asdict(tokenizer_config),
        }
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(config_dict=config_dict)

    model = load_model(
        train_config, fsdp_config, rank=rank if train_config.enable_fsdp else None
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    train_serializer = get_serializer(train_config.serializer_cls)
    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=train_config.model_name,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=train_serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )
    # sanity check for tokenizer
    sanity_check_tokenizer(tokenizer, train_config.model_name)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # TODO(jpgard): below should be deprecated once we verify that bf16
    #  loading works.
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, lora_config.__dict__)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        model = fsdp_wrap_model(model, train_config, fsdp_config, rank)
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

    elif not train_config.quantization and not train_config.enable_fsdp:
        print(
            "[WARNING] applying activation checkpointing by default to non-fsdp model"
        )
        apply_fsdp_checkpointing(model)

        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    #################### Dataset setup ######################

    assert (
        data_arguments.use_preserialized
    ), "only preserialized training data is supported."
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer, use_position_ids=data_arguments.use_position_ids
    )
    train_ds_tokenized = prepare_tokenized_dataset(
        data_arguments=data_arguments,
        train_config=train_config,
        serializer=None,
        accelerator=None,
        tokenizer=tokenizer,
        task_names=train_task_names,
        # TODO: set this as a DataArguments flag or some other flag instead to provide finer-grained control.
        #  Note that setting shuffle=False here will ONLY suppress shuffling of rows before packing;
        #  shards are still shuffled, and samples are shuffled *after* packing.
        #  See load_and_tokenize_preserialized_wds() for more information about how shuffling can be controlled.
        shuffle=False,
        split="train",
    )["train"]

    # Note: do NOT pin memory when using webdataset; leads to weird nccl issues.
    train_dataloader = torch.utils.data.DataLoader(
        train_ds_tokenized,
        batch_size=train_config.batch_size_training,
        collate_fn=data_collator,
        num_workers=train_config.num_workers_dataloader,
    )

    if train_config.run_validation:
        eval_task_names = get_task_names_list(eval_task_file)

        eval_ds_tokenized = prepare_tokenized_dataset(
            data_arguments=data_arguments,
            train_config=train_config,
            serializer=None,
            accelerator=None,
            tokenizer=tokenizer,
            task_names=eval_task_names,
            shuffle=False,
            split="test",
        )["test"]

        # Note: do NOT pin memory when using webdataset; leads to weird nccl issues.
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds_tokenized,
            batch_size=train_config.batch_size_training,
            collate_fn=data_collator,
            num_workers=train_config.num_workers_dataloader,
        )
    else:
        eval_dataloader = None

    #################### Training setup ######################

    # Initialize the optimizer and learning rate scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps
        if train_config.warmup_steps
        else int(train_config.warmup_ratio * train_config.max_steps),
        num_training_steps=train_config.max_steps
        // train_config.gradient_accumulation_steps,
    )

    if train_config.resume:
        ckpt_dir = get_latest_checkpoint(train_config.resume)
        print("#" * 50)
        print(f"Resuming scheduler, optimizer, and model from {ckpt_dir}")
        optimizer_state = torch.load(
            os.path.join(ckpt_dir, OPTIMIZER_STATE_PT), map_location="cpu"
        )
        if train_config.enable_fsdp:
            optimizer_state = FSDP.optim_state_dict_to_load(
                model=model, optim=optimizer, optim_state_dict=optimizer_state
            )
        optimizer.load_state_dict(optimizer_state)
        scheduler_state = torch.load(
            os.path.join(ckpt_dir, SCHEDULER_STATE_PT), map_location="cpu"
        )
        scheduler.load_state_dict(scheduler_state)
        model, global_step = load_model_from_checkpoint(model, ckpt_dir)
        global_step += 1
    else:
        global_step = 0

    if train_config.torch_compile:
        print("compiling with torch.compile()")
        model = torch.compile(
            model,
            fullgraph=train_config.torch_compile_fullgraph,
        )
        print("compiling with torch.compile() complete")

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank=local_rank if torch.distributed.is_initialized() else None,
        rank=rank if torch.distributed.is_initialized() else None,
        wandb_run=wandb_run,
        epoch_length=train_config.max_steps,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (FsdpConfig, LoraConfig, TrainConfig, DataArguments, TokenizerConfig)
    )

    parser.add_argument("--train-task-file", type=str, default=None)
    # parser.add_argument("--train-eval-task-file", type=str, default=None)
    parser.add_argument("--eval-task-file", type=str, default=None)

    (
        fsdp_config,
        lora_config,
        train_config,
        data_arguments,
        tokenizer_config,
        other_args,
    ) = parser.parse_args_into_dataclasses()

    main(
        train_config=train_config,
        fsdp_config=fsdp_config,
        lora_config=lora_config,
        data_arguments=data_arguments,
        tokenizer_config=tokenizer_config,
        **vars(other_args),
    )
