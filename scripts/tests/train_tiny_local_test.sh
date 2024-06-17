#!/bin/bash

# Train from an existing cached dataset.

echo 'activating conda environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rtfm


echo 'running training script'

python scripts/finetune.py \
  --train-task-file "./sampledata/v6.0.3-serialized/test/test-files.txt" \
  --eval-task-file "./sampledata/v6.0.3-serialized/train/train-files.txt" \
  --run_validation "False" \
  --use_wandb "False" \
  --warmup_steps 1 \
  --num_workers_dataloader 0 \
  --batch_size_training 2 \
  --max_steps 10 \
  --model_name "yujiepan/llama-2-tiny-random" \
  --dist_checkpoint_root_folder "checkpoints" \
  --dist_checkpoint_folder "tiny_trainer" \
  --save_model \
  --save_optimizer \
  --serializer_cls "BasicSerializerV2"

# the model is saved at <dist_checkpoint_root_folder>/<dist_checkpoint_folder>-<model_name>
OUTPUT_DIR="checkpoints/tiny_trainer-yujiepan/llama-2-tiny-random/"
echo "got the following output files at ${OUTPUT_DIR}:"
ls $OUTPUT_DIR

echo "removing all files in output dir ${OUTPUT_DIR}"
rm -r $OUTPUT_DIR
