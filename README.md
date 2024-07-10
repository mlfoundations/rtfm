`rtfm` is a Python library for research on tabular foundation models (RTFM).

`rtfm` is the library used to train [TabuLa-8B](https://huggingface.co/mlfoundations/tabula-8b),
a state-of-the-art model for zero- and few-shot tabular data prediction described in our paper
["Large Scale Transfer Learning for Tabular Data via Language Modeling"](https://arxiv.org/abs/2406.12031).

<div align=center>
<img alt="few-shot results curve" src="https://github.com/mlfoundations/rtfm/blob/main/assets/all_tasks_curves.png" width=50%>
</div>

You can also use `rtfm` to train your own tabular language models.

`rtfm` has been used to train 7B- and 8B-parameter Llama 2 and Llama 3 language models,
and supports advanced and efficient training methodologies such as fully sharded data parallel (FSDP),
multinode training, and 16-bit training with `bf16`.
In the future, we plan to support additional base language models and larger scales;
currently, support for larger Llama models exists but should be considered experimental.
We do not currently support other (non-Llama) language models.

# Environment setup

We recommend use of the provided `conda` environment. You can set it up with:

```shell
conda env create -f environment.yml
pip install --no-deps git+https://github.com/mlfoundations/tableshift.git
```
# Quickstart - Inference

If you want to interactively explore the model or want to try it on your own, unlabelled data, the best way to do this is by using the `inference.ipynb` notebook in `notebooks`. This notebook shows how to create simple DataFrames and use them for inference.

The notebook above is our recommended default for users interested in trying out TabuLa-8B. For more fine-grained control over your inference (e.g. changing the system prompt used at inference time), you can use `inference_utils.infer_on_example()`.

# Quickstart - Training

Once you have set up your environment, you can train models using the Python script at `scripts/finetune.py`.
As an example, to conduct a training run with a small toy model, run:

```shell
python scripts/finetune.py \
  --train-task-file "./sampledata/v6.0.3-serialized/test/test-files.txt" \
  --eval-task-file "./sampledata/v6.0.3-serialized/train/train-files.txt" \
  --run_validation "False" \
  --use_wandb "False" \
  --warmup_steps 1 \
  --num_workers_dataloader 0 \
  --max_steps 10 \
  --model_name "yujiepan/llama-2-tiny-random" \
  --dist_checkpoint_root_folder "checkpoints" \
  --dist_checkpoint_folder "my_model_dir" \
  --save_model \
  --save_optimizer
```

This will conduct a short training run and save the model and optimizer state to
`checkpoints/my_model_dir-llama-2-tiny-random`.
To train a Llama3-8B model instead of the toy model in this example, replace `yujiepan/llama-2-tiny-random`
with `meta-llama/Meta-Llama-3-8B`.

# Quickstart - Evaluation

To evaluate a model, we recommend using the Python script at `scripts/evaluate_checkpoint.py`.
In order to evaluate against any dataset, we recommend first preparing the dataset using
the provided script `scripts/utils/prepare_csv_for_eval.py`:

```shell
python scripts/utils/prepare_csv_for_eval.py --output_dir ./eval_tasks/my_task
```

`prepare_csv_for_eval.py` prescribes how the data in the CSV is serialized at evaluation time
by defining a FeatureList and YAML file describing the
features and prediction task, respectively. By default, `prepare_csv_for_eval.py` processes data the same way as the
evaluations for TabuLa-8B. If desired, you can write a custom script to control
the creation of the FeatureList and YAML files to change how data is serialized.

Once the data is prepared, run evaluation via:

```shell
USER_CONFIG_DIR=./eval_tasks/ \
  python scripts/evaluate_checkpoint.py \
  --eval-task-names "my_task" \
  --model_name "yujiepan/llama-2-tiny-random" \
  --resume "checkpoints/my_model_dir-llama-2-tiny-random" \
  --eval_max_samples 16 \
  --context_length 2048 \
  --pack_samples "False" \
  --num_shots 1 \
  --outfile "tmp.csv"
```

This will write a file to `tmp.csv` containing evaluation results.

If you want to evaluate the pretrained released TabuLa-8B model,
set `--model_name` to `mlfoundations/tabula-8b` and remove the `--resume` flag or set it to the empty string.

# Environment Setup and Test

You can create an environment to reproduce or run experiments via `rtfm` by using conda:

```
conda env create -f environment.yml
```

Once you've set up your environment, you need to add your Hugging Face token in order to access the LLama weights. To do
this, you can run

```shell
huggingface-cli login
```

or manually set the token via

```shell
export HF_TOKEN=your_token
```

To test your setup (on any machine, no GPUs required), you can run the following command:

```shell
sh scripts/tests/train_tiny_local_test.sh
```

# End-To-End Training Example

This section gives an example of how to train a model from a set of parquet files.

## 1. Prepare training data.

The model expects sets of serialized records stored in .tar files, which are in webdataset format.
To serialize data, we provide the script `serialize_interleave_and_shuffle.py` (located at rtfm/pipelines/serialize_interleave_and_shuffle.py)
to serialize a set of parquet files:

```shell
python -m rtfm.pipelines.serialize_interleave_and_shuffle \
    --input-dir /glob/containing/parquet/files/ \
    --output-dir ./serialized/v6.0.3/ \
    --max_tables 64 \
    --serializer_cls "BasicSerializerV2"
```

The recommended way to store training data is in a newline-delimited list of webdataset files.
The above command will automatically generate sets of training, validation (`train-eval`), and test
files, where the `train-eval` split comprises unseen rows from tables in the training split,
and the `test` split comprises only unseen tables.

### Using data hosted on S3 (recommended)

Some datasets may be too large to store on disk during training.
`rtfm` supports using files stored on AWS S3.
To use files hosted on S3, you need to move the training data there, and update the text files produced
by `rtfm/pipelines/serialize_interleave_and_shuffle.py` to point to the correct location.
You can do this with `sed`, for example,
the command below will replace the local training location with the s3 path for all lines in a text file:

```shell
sed 's|/path/to/sampledata/|s3://rtfm-hub/tablib/serialized/v0-testdata/|g' traineval-files.txt > traineval-files-s3.txt
```

### Using local training data

If you plan to use local data, you can use the files produced as the output
of `serialize_interleave_and_shuffle.py` (`train-files.txt`, `traineval-files.txt`, `test-files.txt`).

## 2. Launch a training job.

The recommended way to launch a training job is via `finetune.py`. You can do this, for example, via:

```shell
python scripts/finetune.py \
  --train-task-file "./sampledata/v6.0.3-serialized/test/test-files.txt" \
  --eval-task-file "./sampledata/v6.0.3-serialized/train/train-files.txt" \
  --run_validation "False" \
  --use_wandb "False" \
  --warmup_steps 1 \
  --num_workers_dataloader 0 \
  --max_steps 10 \
  --model_name "yujiepan/llama-2-tiny-random" \
  --dist_checkpoint_root_folder "checkpoints" \
  --dist_checkpoint_folder "my_model_dir" \
  --save_model \
  --save_optimizer
```

See `finetune.py` and the associated configuration classes in `rtfm.configs` and `rtfm.arguments`
for more options to control the details of training.

# Additional Resources

Some additional resources relevant to RTFM:

* Our paper, ["Large Scale Transfer Learning for Tabular Data via Language Modeling"](https://arxiv.org/abs/2406.12031)
* The [t4 dataset](https://huggingface.co/datasets/mlfoundations/t4-full) on Hugging Face (used to train TabuLa-8B)
* The TabuLa-8B [evaluation suite data](https://huggingface.co/datasets/mlfoundations/tabula-8b-eval-suite) on Hugging
  Face
* The [TabuLa-8B model](https://huggingface.co/mlfoundations/tabula-8b) on Hugging Face
* [`tabliblib`](https://github.com/mlfoundations/tabliblib), a toolkit for filtering TabLib
