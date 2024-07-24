#!/bin/bash

# Train from an existing cached dataset.

echo 'activating conda environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rtfm


echo 'running training script'

python -m rtfm.finetune \
  --train-task-file "tablib-sample/train-files.txt" \
  --eval-task-file "tablib-sample/eval-files.txt" \
  --run_validation "False" \
  --use_wandb "False" \
  --warmup_steps 4 \
  --num_workers_dataloader 8 \
  --max_steps 16 \
  --save_checkpoint_root_dir "checkpoints" \
  --run_name "fsdp_test" \
  --save_model \
  --save_optimizer \
  --enable_fsdp \
  --pure_bf16 \
  --use_fast_kernels \
  --batch_size_training 8

