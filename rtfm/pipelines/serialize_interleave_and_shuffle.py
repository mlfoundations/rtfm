"""
Serialize and interleave a set of datasets. This goes from a
set of individual parquet files, each representing a dataset,
to a set of parquet files where each element is a serialized observation
from one of those datasets. Those observations can be tokenized
and used for training directly, instead of needing to be serialized first.

Usage:
python -m rtfm.pipelines.serialize_interleave_and_shuffle \
    --input_dir "/path/or/wildcard/to/parquet/directories/" \
    --output_dir /path/to/output/ \
    --chunk_size 256 \
    --max_tables 100_000
"""
import glob
import json
import logging
import random
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Optional

import pandas as pd
import ray
from sklearn.model_selection import train_test_split
from transformers import HfArgumentParser

from rtfm.arguments import DataArguments
from rtfm.configs import SerializerConfig
from rtfm.data import (
    build_formatted_df_from_file,
    example_map_fn,
    NoTargetCandidatesError,
)
from rtfm.serialization.serializers import get_serializer


def process_file(
    row,
    data_args: DataArguments,
    serializer_config: SerializerConfig,
    model_max_len_tokens=4096,
    appx_chars_per_token=3.5,
):
    filename = row["item"]
    logging.warning(f"loading {filename}")

    try:
        df = build_formatted_df_from_file(
            filename,
            data_args=data_args,
        )
    except NoTargetCandidatesError:
        return {"item": "Failed"}
    except ValueError as ve:
        logging.error(ve)
        return {"item": "Failed"}
    except TypeError as te:
        logging.error(te)
        return {"item": "Failed"}

    records = df.to_dict(orient="records")

    serializer = get_serializer(serializer_config)
    _map_fn = partial(
        example_map_fn,
        data_args=data_args,
        serializer=serializer,
        cfg=None,
    )

    for record in records:
        mapped = _map_fn(record)
        # Do not keep examples that would not fit in the model's context
        if len(mapped["text"]) > int(model_max_len_tokens * appx_chars_per_token):
            logging.warning(
                f"dropping too-long sample with text len {len(mapped['text'])}"
            )
            continue
        # after applying map_fn, each element has fields: 'text', 'class_label_as_text'
        yield {**mapped, "filename": filename}


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


import os
import shutil
import tempfile
import webdataset as wds
from tqdm import tqdm


@dataclass
class Resharder:
    target_size_mb: int
    extension: str = ".tar"

    @property
    def target_size(self):
        return self.target_size_mb * 1024 * 1024  # Convert MB to bytes

    def reshard(self, dirname, prefix: Optional[str] = None):
        # Get all self.extension files in the directory
        input_shards = [f for f in os.listdir(dirname) if f.endswith(self.extension)]
        input_shards.sort()  # Ensure consistent ordering

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a ShardWriter with the target size
            filename = (
                "-".join([x for x in (prefix, "%06d") if x is not None])
                + self.extension
            )
            writer = wds.ShardWriter(
                pattern=os.path.join(temp_dir, filename),
                maxsize=self.target_size,
            )

            # Iterate through all input shards
            for shard in tqdm(
                input_shards, desc=f"Resharding {os.path.basename(dirname)}"
            ):
                # Open each shard and write its contents to the new shards
                with wds.WebDataset(os.path.join(dirname, shard)) as dataset:
                    for sample in dataset:
                        writer.write(sample)

            # Close the writer to ensure all data is written
            writer.close()

            # Delete original shards
            for shard in input_shards:
                os.remove(os.path.join(dirname, shard))

            # Move new shards from temp directory to original directory
            for shard in os.listdir(temp_dir):
                shutil.move(os.path.join(temp_dir, shard), os.path.join(dirname, shard))

        print(f"Resharding complete for {dirname}")

    def reshard_all_subdirectories(self, root_dir, prefix):
        # Iterate over all items in the root directory
        for split in os.listdir(root_dir):
            # Construct full path
            item_path = os.path.join(root_dir, split)

            # Check if it's a directory
            if os.path.isdir(item_path):
                print(f"Processing subdirectory: {split}")

                # Check if the subdirectory contains any self.extension files
                if any(file.endswith(self.extension) for file in os.listdir(item_path)):
                    # Apply reshard function to this subdirectory
                    self.reshard(item_path, "-".join([x for x in (prefix, split) if x]))
                else:
                    print(f"Skipping {split}: No {self.extension} files found")


def convert_to_wds(
    parquet_files: Sequence[str],
    index: int,
    output_dir: str,
    prefix: str,
    split: str,
    eval_split=0.01,
):
    do_eval_split = split == "train" and random.uniform(0.0, 1.0) < eval_split

    base_file_name = "-".join([x for x in (prefix, split, f"{index:06d}.tar") if x])
    # Output WebDataset file name based on the parquet file name
    wds_filename = os.path.join(output_dir, split, base_file_name)

    sink = wds.TarWriter(wds_filename)
    if do_eval_split:
        wds_eval_filename = os.path.join(output_dir, split + "eval", base_file_name)
        os.makedirs(os.path.dirname(wds_eval_filename), exist_ok=True)
        eval_sink = wds.TarWriter(wds_eval_filename) if do_eval_split else None

    # Open a WebDataset writer
    def encode_row(ser: pd.Series):
        return json.dumps(ser.to_dict(), ensure_ascii=False).encode("utf-8")

    for parquet_file in parquet_files:
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        # Extract filename without extension for use in webdataset keys
        base_filename = os.path.splitext(os.path.basename(parquet_file))[0]

        # Randomly split the DataFrame into train and eval if this is train_eval split
        if do_eval_split:
            # Try/Except to handle small dataframes that cannot be split
            try:
                train_df, eval_df = train_test_split(
                    df, test_size=eval_split, random_state=42
                )
            except ValueError:
                logging.warning(f"Skipping train_eval split of df with size {len(df)}")
                train_df = df
                eval_df = None
        else:
            train_df = df
            eval_df = None

        # Process the training data
        for index, row in train_df.iterrows():
            key = f"{base_filename}__{index}"
            sink.write({"__key__": key, "json": encode_row(row)})

        # Process the evaluation data if applicable
        if do_eval_split and (eval_df is not None):
            for index, row in eval_df.iterrows():
                key = f"{base_filename}__{index}"
                eval_sink.write({"__key__": key, "json": encode_row(row)})

        os.remove(parquet_file)

    # Close the WebDataset writers
    sink.close()
    if do_eval_split and eval_sink:
        eval_sink.close()


def parquet_to_wds(
    parquet_files,
    prefix: str,
    split: str,
    chunk_size: int,
    target_shard_size_mb: int,
    output_dir="output",
):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Chunk the files
    file_chunks = list(chunked(parquet_files, chunk_size))

    # Create a pool of workers
    with Pool(os.cpu_count()) as pool:
        # Map parquet files to the converter function with the specified output directory
        list(
            tqdm(
                pool.starmap(
                    convert_to_wds,
                    [
                        (file_chunk, i, output_dir, prefix, split)
                        for i, file_chunk in enumerate(file_chunks)
                    ],
                ),
                total=len(parquet_files) // chunk_size,
                desc=f"{prefix} parquet to wds",
            )
        )

    # reshard the outputs
    resharder = Resharder(target_shard_size_mb)
    resharder.reshard_all_subdirectories(output_dir, prefix)


@dataclass
class PipelineConfig:
    input_dir: str
    output_dir: str
    max_tables: Optional[int] = None
    train_frac: float = 0.975
    split_random_seed = 42
    output_shard_factor: int = 1000
    output_file_prefix: Optional[str] = None
    chunk_size: int = 64
    target_shard_size_mb: int = 500


def main(
    serializer_config: SerializerConfig,
    data_args: DataArguments,
    pipeline_config: PipelineConfig,
):
    data_args.use_config = False
    data_args.feature_name_handling = "none"
    data_args.feature_value_handling = "none"
    data_args.targets_handling = "none"

    if pipeline_config.max_tables:
        logging.warning(f"pipeline_config.max_tables is {pipeline_config.max_tables}")
    print(f"ray version is {ray.__version__}")
    start = time.time()
    files = glob.glob(os.path.join(pipeline_config.input_dir, "*.parquet"))
    print(f"got {len(files)} parquet files")

    if (
        pipeline_config.max_tables is not None
        and len(files) > pipeline_config.max_tables
    ):
        files = files[: pipeline_config.max_tables]
    print(f"files is {files}")
    ray.init(address="auto")

    num_nodes = len(ray.nodes())
    print(f"num nodes = {num_nodes}")
    num_cores = os.cpu_count()
    print(f"num cores = {num_cores}")
    parallelism = num_nodes * num_cores
    print(f"parallelism = {parallelism}")

    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = True
    ctx.max_errored_blocks = 1000

    # Fully shuffle the input files.
    train_files, split_files = train_test_split(
        files,
        test_size=1 - pipeline_config.train_frac,
        random_state=pipeline_config.split_random_seed,
    )
    train_ds = ray.data.from_items(train_files)
    test_ds = ray.data.from_items(split_files)

    # Repartition to balance the size of shards (smaller shards help avoid OOM),
    # and also to control the size of the output files (this keeps output files small, which helps
    # us shuffle them later).

    fn_kwargs = {
        "data_args": data_args,
        "serializer_config": serializer_config,
    }
    test_ds = test_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * pipeline_config.output_shard_factor
    )
    train_ds = train_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * pipeline_config.output_shard_factor
    )

    splits = ("train", "test")

    for ds, split in zip((train_ds, test_ds), splits):
        ds.write_parquet(
            f"local://{os.path.abspath(pipeline_config.output_dir)}/{split}"
        )

    ray.shutdown()

    print(
        f"finished ray pipeline in {time.time() - start} secs. Files are written to {pipeline_config.output_dir}"
    )

    for split in ("test", "train"):
        split_files = glob.glob(
            os.path.join(pipeline_config.output_dir, split, "*.parquet")
        )
        print(f"converting {len(split_files)} files to webdataset for split {split}.")
        parquet_to_wds(
            split_files,
            prefix=pipeline_config.output_file_prefix
            if pipeline_config.output_file_prefix is not None
            else "",
            split=split,
            chunk_size=pipeline_config.chunk_size,
            output_dir=os.path.join(pipeline_config.output_dir),
            target_shard_size_mb=pipeline_config.target_shard_size_mb,
        )

    return


if __name__ == "__main__":
    parser = HfArgumentParser((SerializerConfig, DataArguments, PipelineConfig))
    (
        serializer_config,
        data_args,
        pipeline_config,
    ) = parser.parse_args_into_dataclasses()
    main(serializer_config, data_args, pipeline_config)
