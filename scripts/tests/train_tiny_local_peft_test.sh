#!/bin/bash

# Train from an existing cached dataset.

echo 'activating conda environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rtfm


echo 'running training script'

python scripts/finetune.py \
  --train-task-file "tablib-sample/train-files.txt" \
  --eval-task-file "tablib-sample/eval-files.txt" \
  --run_validation "False" \
  --use_wandb "False" \
  --warmup_steps 1 \
  --num_workers_dataloader 0 \
  --batch_size_training 2 \
  --max_steps 10 \
  --use_peft \
  --peft_method "lora" \
  --r 8 \
  --lora_alpha 32 \
  --model_name "yujiepan/llama-2-tiny-random" \
  --save_checkpoint_root_dir "checkpoints" \
  --run_name "tiny_trainer" \
  --save_model \
  --save_optimizer

# the model is saved at <save_checkpoint_root_dir>/<run_name>-<model_name>
OUTPUT_DIR="checkpoints/tiny_trainer-yujiepan/llama-2-tiny-random/"
echo "got the following output files at ${OUTPUT_DIR}:"
ls $OUTPUT_DIR

echo "removing all files in output dir ${OUTPUT_DIR}"
rm -r $OUTPUT_DIR
