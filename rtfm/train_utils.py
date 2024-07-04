import glob
import logging
import os
import re
import shutil
import time
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from typing import Union, Optional, Dict, Tuple

import boto3
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.fsdp as FSDP
import transformers
from accelerate.utils import is_xpu_available
from botocore.exceptions import ClientError
from llama_recipes.model_checkpointing.checkpoint_handler import (
    fullstate_save_policy,
)
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import save_train_params
from safetensors import safe_open
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from rtfm.configs import TrainConfig
from rtfm.utils import timestamp

SCALER_STATE_PT = "scaler_state.pt"
MODEL_STATE_PT = "model.pt"
SCHEDULER_STATE_PT = "scheduler_state.pt"
OPTIMIZER_STATE_PT = "optimizer_state.pt"


def load_optimizer_from_checkpoint(
    model, optimizer, ckpt_dir, train_config: TrainConfig, rank
):
    optimizer_pt = os.path.join(ckpt_dir, OPTIMIZER_STATE_PT)

    print(f"loading optimizer state from {optimizer_pt} on rank {rank}...")
    optimizer_state = torch.load(optimizer_pt, map_location="cpu")
    if train_config.enable_fsdp:
        # Load optimizer state dict, following example usage in
        # torch.distributed.fsdp.fully_sharded_data_parallel.FSDP.optim_state_dict()

        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        )

        optim_state_dict = FSDP.optim_state_dict_to_load(
            model, optimizer, optimizer_state
        )
        optimizer.load_state_dict(optim_state_dict)

    else:
        optimizer.load_state_dict(optimizer_state)
    print(f"finished loading optimizer state from {optimizer_pt} on rank {rank}")
    return optimizer


def load_model_from_checkpoint(
    model, ckpt_dir, train_config: TrainConfig, rank
) -> Tuple[torch.nn.Module, int]:
    print(f"loading model weights from {ckpt_dir} on rank {rank}...")

    # Initialize an empty state dictionary
    state_dict = {}

    model_state_file = os.path.join(ckpt_dir, MODEL_STATE_PT)
    if os.path.exists(model_state_file):
        print(f"loading model from {model_state_file}")
        state_dict = torch.load(model_state_file, map_location="cpu")
    else:
        shard_files = glob.glob(os.path.join(ckpt_dir, "*.safetensors"))
        print(f"loading model from files {shard_files}")
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in tqdm(f.keys(), desc=f"load from {shard_file}"):
                    state_dict[key] = f.get_tensor(key)
    if train_config.enable_fsdp:
        # Configure FSDP to use full state dict for model
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        # Load model state dict
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            model.load_state_dict(state_dict)

    else:
        model.load_state_dict(state_dict)

    print(f"finished loading model weights from {ckpt_dir} on rank {rank}")
    step = re.search("step-(\d+)", ckpt_dir).group(1)
    print(f"loaded step is {step}")

    return model, int(step)


def make_save_folder_name(cfg: TrainConfig, step: Optional[int] = None) -> str:
    """Encapsulates the llama-recipes logic for building a folder name.

    Instead of duplicating this code in many places, we implement it in a single reusable function here.

    If step is not provided, this function returns the parent checkpoint directory for the TrainConfig.

    If step is provided, this function returns the directory for a specific step (a subdirectory inside the parent directory).
    """
    base_dir = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name.split("/")[-1]
    )
    if not step:
        return base_dir
    else:
        return base_dir + "/" + f"step-{step}"


def save_state_dict_to_default_directory(
    state_dict: Dict[str, torch.Tensor], cfg: TrainConfig, step: int, filename: str
):
    """Save an arbitrary state dict using a consistent path and file naming schema."""
    save_dir = make_save_folder_name(cfg, step)
    os.makedirs(save_dir, exist_ok=True)

    save_full_path = os.path.join(save_dir, filename)

    # Handle case where cfg.model_name contains a slash (i.e. uses a hf repo)
    os.makedirs(os.path.dirname(save_full_path), exist_ok=True)

    print(f"--> saving state to {save_full_path}")

    torch.save(state_dict, save_full_path)

    print(f"--> finished saving to {save_full_path}")
    return


def save_model_and_optimizer_unsharded(
    model,
    optimizer,
    lr_scheduler,
    rank,
    cfg: TrainConfig,
    step: int,
):
    """Saving model via rank0 cpu streaming and full_state_dict, if FSDP is used."""

    # create save path
    save_dir = make_save_folder_name(cfg, step)
    os.makedirs(save_dir, exist_ok=True)

    optim_state = None

    # FSDP model saving

    if cfg.enable_fsdp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = model.state_dict()

            print(f"saving process: rank {rank}  done w model state_dict\n")

    if cfg.enable_fsdp and cfg.save_optimizer:
        optim_state = FSDP.full_optim_state_dict(model, optimizer)

    if cfg.enable_fsdp and rank == 0:
        print(f"--> saving FSDP model on rank 0...")
        save_state_dict_to_default_directory(cpu_state, cfg, step, MODEL_STATE_PT)

    # non-FSDP model saving
    elif not cfg.enable_fsdp:
        model.save_pretrained(save_dir)
        print(f"HF model checkpoint saved for step {step} at {save_dir}\n")
        if cfg.save_optimizer:
            optim_state = optimizer.state_dict()

    # optimizer and scheduler saving
    if cfg.save_optimizer and ((not cfg.enable_fsdp) or rank == 0):
        assert (
            optim_state is not None
        ), f"expected optimizer state; could be unhandled case."
        print(f"--> saving optimizer state...")
        save_state_dict_to_default_directory(optim_state, cfg, step, OPTIMIZER_STATE_PT)

        print(f"--> saving scheduler state...")
        scheduler_state = lr_scheduler.state_dict()
        save_state_dict_to_default_directory(
            scheduler_state, cfg, step, SCHEDULER_STATE_PT
        )


def save_train_state(
    train_config: TrainConfig,
    model,
    optimizer,
    lr_scheduler,
    rank,
    fsdp_config,
    step: int,
):
    """Save the model, optimizer, scheduler, and other info to restore the training state.

    Saving is conducted as specified in the TrainConfig (e.g. full vs. sharded state dict,
    optional saving of optimizer state).
    """
    checkpoint_start_time = time.perf_counter()
    is_main_process = (not train_config.enable_fsdp) or (
        train_config.enable_fsdp and rank == 0
    )
    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.use_peft:
        model.save_pretrained(train_config.output_dir)
        if is_main_process:
            print(f"PEFT modules are saved in {train_config.output_dir} directory")

    else:
        if not train_config.use_peft and not train_config.enable_fsdp:
            save_model_and_optimizer_unsharded(
                model, optimizer, lr_scheduler, rank, train_config, step=step
            )

        elif (
            not train_config.use_peft
            and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT
        ):
            save_model_and_optimizer_unsharded(
                model, optimizer, lr_scheduler, rank, train_config, step=step
            )

        else:
            raise NotImplementedError(
                "State dict saving for the current training configuration not implemented."
            )

    if train_config.enable_fsdp:
        dist.barrier()

    # Remove checkpoints if too many have accumulated.
    if train_config.save_total_limit and is_main_process:
        save_dir = make_save_folder_name(train_config)
        ckpt_dirs = [x for x in glob.glob(os.path.join(save_dir, "*step*"))]

        # Sort oldest-first
        ckpt_dirs = sorted(ckpt_dirs, key=os.path.getmtime)

        if len(ckpt_dirs) > train_config.save_total_limit:
            num_to_remove = len(ckpt_dirs) - train_config.save_total_limit
            for dir_to_remove in ckpt_dirs[:num_to_remove]:
                shutil.rmtree(dir_to_remove)

    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
    return checkpoint_end_time


def train(
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    train_config: TrainConfig,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    wandb_run=None,
    epoch_length=None,
    step: int = 0,
):
    """
    Trains the model on the given dataloader

    Adapted from llama-recipes/utils/train_utils.py.

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    gradient_accumulation_steps = train_config.gradient_accumulation_steps
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()

    if not epoch_length:
        epoch_length = len(train_dataloader)

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    model.train()
    total_length = epoch_length // gradient_accumulation_steps
    pbar = tqdm(
        colour="blue",
        desc=f"Train",
        initial=step,
        total=total_length,
        dynamic_ncols=True,
    )
    for batch in train_dataloader:
        model.train()
        start_ts = timestamp()
        for key in batch.keys():
            if not torch.cuda.is_available():
                pass
            elif train_config.enable_fsdp:
                if is_xpu_available():
                    batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                else:
                    batch[key] = batch[key].to(local_rank)
            else:
                if is_xpu_available():
                    batch[key] = batch[key].to("xpu:{local_rank}")
                else:
                    batch[key] = batch[key].to(f"cuda:{local_rank}")
        with autocast():
            loss = model(**batch).loss
        loss = loss / gradient_accumulation_steps

        if train_config.use_fp16:
            # if fp16 is enabled, use gradient scaler to handle gradient update
            scaler.scale(loss).backward()
            if (
                step + 1
            ) % gradient_accumulation_steps == 0 or step == epoch_length - 1:
                if (
                    train_config.gradient_clipping
                    and train_config.gradient_clipping_threshold > 0.0
                ):
                    scaler.unscale_(optimizer)
                    if train_config.enable_fsdp:
                        model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_config.gradient_clipping_threshold,
                        )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                pbar.update(1)
        else:
            # regular backpropagation when fp16 is not used
            loss.backward()
            if (
                step + 1
            ) % gradient_accumulation_steps == 0 or step == epoch_length - 1:
                if (
                    train_config.gradient_clipping
                    and train_config.gradient_clipping_threshold > 0.0
                ):
                    if train_config.enable_fsdp:
                        model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            train_config.gradient_clipping_threshold,
                        )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                pbar.update(1)

        if wandb_run:
            if not train_config.enable_fsdp or rank == 0:
                batch_tokens_count = np.prod(list(batch["input_ids"].shape))
                step_time = timestamp() - start_ts
                wandb_run.log(
                    {
                        "train/step": step,
                        "train/loss": loss.detach().float(),
                        "train/perplexity": float(torch.exp(loss.detach().float())),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/tokens_per_batch": batch_tokens_count,
                        "train/step_time_secs": step_time,
                        "train/tokens_per_gpu_per_sec": batch_tokens_count / step_time,
                    },
                    step=step,
                )

        if train_config.save_steps and (step + 1) % train_config.save_steps == 0:
            checkpoint_end_time = save_train_state(
                train_config,
                model,
                optimizer,
                lr_scheduler,
                rank,
                fsdp_config,
                step,
            )
            if wandb_run:
                wandb_run.log({"train/checkpoint_time": checkpoint_end_time}, step=step)

        pbar.set_description(f"Step {step} loss: {loss.detach().float()}")

        if train_config.run_validation and (step + 1) % train_config.eval_steps == 0:
            evaluate(
                model,
                train_config,
                eval_dataloader,
                local_rank,
                step,
                wandb_run,
                max_batches=train_config.eval_max_batches,
            )

        step += 1
        if step >= epoch_length - 1:
            break

    pbar.close()

    # TODO(jpgard): log results here; also log the validation results below.

    if not train_config.enable_fsdp or rank == 0:
        memtrace.print_stats()

    if train_config.save_model:
        checkpoint_end_time = save_train_state(
            train_config, model, optimizer, lr_scheduler, rank, fsdp_config, step
        )

    if train_config.run_validation:
        evaluate(
            model,
            train_config,
            eval_dataloader,
            local_rank,
            step,
            wandb_run,
            max_batches=train_config.eval_max_batches,
        )

    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return


def evaluate(
    model,
    train_config: TrainConfig,
    eval_dataloader: torch.utils.data.DataLoader,
    local_rank: int,
    step: int,
    wandb_run=None,
    max_batches: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the model on the given dataloader

    Adapted from llama-recipes/utils/train_utils.py.


    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []

    eval_loss_and_samples = torch.zeros((2,), dtype=torch.float).to(
        local_rank
    )  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for idx, batch in enumerate(
            tqdm(
                eval_dataloader,
                colour="green",
                desc="evaluating Epoch",
                dynamic_ncols=True,
            )
        ):
            for key in batch.keys():
                if not torch.cuda.is_available():
                    pass
                elif train_config.enable_fsdp:
                    if is_xpu_available():
                        batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                    else:
                        batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to("xpu:{local_rank}")
                    else:
                        batch[key] = batch[key].to(f"cuda:{local_rank}")

            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss

                eval_loss_and_samples[0] += loss.detach().float()
                eval_loss_and_samples[1] += len(batch["input_ids"])

            if idx >= max_batches:
                print(f"terminating eval after {idx} steps")
                break

    # If there's more than one CUDA device, reduce evaluation loss
    # and sample count across all devices
    if is_xpu_available() and (
        torch.xpu.device_count() > 1 and train_config.enable_fsdp
    ):
        dist.all_reduce(eval_loss_and_samples, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss_and_samples, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_loss = eval_loss_and_samples[0]
    samples_seen = eval_loss_and_samples[1]
    eval_epoch_loss = eval_loss / samples_seen
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    eval_metrics = {
        "eval/perplexity": eval_ppl.item(),
        "eval/loss": eval_epoch_loss.item(),
    }
    if wandb_run:
        wandb_run.log(eval_metrics, step=step)
    else:
        print(f"eval metrics at step {step}: {eval_metrics}")

    return eval_ppl, eval_epoch_loss


def parse_s3_filename(fp):
    assert fp.startswith("s3://")
    bucket, dirpath = fp.replace("s3://", "").split("/", maxsplit=1)
    return bucket, dirpath


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    if not os.path.exists(file_name):
        raise ValueError(
            f"File {file_name} does not exist. Did you provide the complete path?"
        )
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    logging.warning(f"writing {file_name} to {bucket}/{object_name}")

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def save_pretrained(
    model_or_tokenizer: Union[
        transformers.LlamaForCausalLM, transformers.PreTrainedTokenizer
    ],
    output_dir: str,
    is_main_process: bool = True,
    state_dict: Optional[dict] = None,
):
    """Wrapper around the hugging face .save_pretrained() method to allow saving to s3.

    Note that when saving to s3 and not a local file path,
     this function first creates a local copy of the file, then uploads to s3.
    """
    logging.warning(f"saving pretrained model or tokenizer to {output_dir}")
    # If output_dir is an s3 path, save to a local tmpdir, then upload to s3
    if output_dir.startswith("s3://"):
        with TemporaryDirectory() as tmpdir:
            model_or_tokenizer.save_pretrained(
                tmpdir,
                is_main_process=is_main_process,
                state_dict=state_dict,
            )
            bucket, dirpath = parse_s3_filename(output_dir)
            for local_tmp_file in os.listdir(tmpdir):
                object_name = os.path.join(dirpath, local_tmp_file)
                upload_file(os.path.join(tmpdir, local_tmp_file), bucket, object_name)

    else:
        model_or_tokenizer.save_pretrained(
            output_dir,
            is_main_process=is_main_process,
            state_dict=state_dict,
        )
