#!/bin/bash

echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate rtfm

# Note: this script evaluates a pretrained (*not* fine-tuned) model, as an integration
# test of the eval loop. If you want to test the (fine-tune --> evaluate) loop, then
# use the file train_and_eval_tiny_local_test.sh .
USER_CONFIG_DIR="./sampledata" \
python scripts/evaluate_checkpoint.py \
  --eval-task-names "dummy_n10000_d4_numclasses4" \
  --model_name "yujiepan/llama-2-tiny-random" \
  --eval_max_samples 128 \
  --context_length 2048 \
  --feature_value_handling "map" \
  --feature_name_handling "map" \
  --pack_samples "False" \
  --num_shots 1 \
  --output_dir "checkpoints/tiny_trainer" \
  --outfile "tmp.csv"

echo "removing outfile tmp.csv"
rm "tmp.csv"