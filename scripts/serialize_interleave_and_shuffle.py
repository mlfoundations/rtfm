"""
Serialize and interleave a set of datasets. This goes from a
set of individual parquet files, each representing a dataset,
to a set of parquet files where each element is a serialized observation
from one of those datasets. Those observations can be tokenized
and used for training directly, instead of needing to be serialized first.

Usage:
python scripts/serialize_interleave_and_shuffle.py \
    --input-dir "/path/or/wildcard/to/parquet/chunk-001[2-4]/" \
    --input-dir /path/to/output/ \
    --chunk_size 256 \
    --max_tables 100_000
"""
import glob
import json
import logging
import os
import random
import time
from functools import partial
from multiprocessing import Pool
from typing import Sequence, Optional

import fire
import pandas as pd
import ray
import webdataset as wds
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from rtfm.arguments import DataArguments
from rtfm.data import (
    build_formatted_df_from_file,
    example_map_fn,
    NoTargetCandidatesError,
)
from rtfm.serialization.serializers import get_serializer


def write_file_list(files, output: str) -> None:
    print(f"writing list of {len(files)} files to {output}")
    with open(output, "w") as f:
        for i, file in enumerate(files):
            if i + 1 < len(files):
                f.write(os.path.abspath(file) + "\n")
            else:
                f.write(os.path.abspath(file))
    return


def process_file(
    row,
    data_args: DataArguments,
    model_max_len_tokens=4096,
    appx_chars_per_token=3.5,
    serializer_cls="BasicSerializerV2",
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

    serializer = get_serializer(serializer_cls)
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


def convert_to_wds(
    parquet_files: Sequence[str],
    index: int,
    output_dir: str,
    prefix: str,
    eval_split=0.1,
):
    do_train_eval = prefix == "train" and random.uniform(0.0, 1.0) > 0.9
    # Output WebDataset file name based on the parquet file name
    wds_filename = os.path.join(output_dir, f"{prefix}-{index:06d}.tar")
    wds_eval_filename = os.path.join(output_dir, f"{prefix}eval-{index:06d}.tar")

    sink = wds.TarWriter(wds_filename)
    eval_sink = wds.TarWriter(wds_eval_filename) if do_train_eval else None

    # Open a WebDataset writer
    def encode_row(ser: pd.Series):
        return json.dumps(ser.to_dict(), ensure_ascii=False).encode("utf-8")

    for parquet_file in parquet_files:
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        # Extract filename without extension for use in webdataset keys
        base_filename = os.path.splitext(os.path.basename(parquet_file))[0]

        # Randomly split the DataFrame into train and eval if this is train_eval split
        if do_train_eval:
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
        if eval_sink and (eval_df is not None):
            for index, row in eval_df.iterrows():
                key = f"{base_filename}__{index}"
                eval_sink.write({"__key__": key, "json": encode_row(row)})

        os.remove(parquet_file)

    # Close the WebDataset writers
    sink.close()
    if eval_sink:
        eval_sink.close()


def parquet_to_wds(parquet_files, prefix: str, chunk_size, output_dir="output"):
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
                        (file_chunk, i, output_dir, prefix)
                        for i, file_chunk in enumerate(file_chunks)
                    ],
                ),
                total=len(parquet_files) // chunk_size,
                desc=f"{prefix} parquet to wds",
            )
        )


def main(
    input_dir: str,
    serializer_cls: str = "BasicSerializerV2",
        output_dir="./sampledata",
    max_tables: Optional[int] = None,
    train_frac: float = 0.975,
    split_random_seed=42,
    output_shard_factor: int = 1000,
    output_file_prefix: Optional[str] = None,
    chunk_size: int = 64,
    labels_drop_numeric: bool = False,
    labels_p_numeric: float = 0.1,
):
    if max_tables:
        logging.warning(f"max_tables is {max_tables}")
    print(f"ray version is {ray.__version__}")
    start = time.time()
    files = glob.glob(os.path.join(input_dir, "*.parquet"))
    print(f"got {len(files)} parquet files")

    if max_tables is not None and len(files) > max_tables:
        files = files[:max_tables]
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
        files, test_size=1 - train_frac, random_state=split_random_seed
    )
    train_ds = ray.data.from_items(train_files)
    test_ds = ray.data.from_items(split_files)

    # Repartition to balance the size of shards (smaller shards help avoid OOM),
    # and also to control the size of the output files (this keeps output files small, which helps
    # us shuffle them later).
    data_args = DataArguments(
        targets_handling="none",
        feature_name_handling="none",
        feature_value_handling="none",
        use_config=False,
        use_metafeatures=False,
        use_task_context=False,
        labels_drop_numeric=labels_drop_numeric,
        labels_p_numeric=labels_p_numeric,
    )
    fn_kwargs = {
        "data_args": data_args,
        "serializer_cls": serializer_cls,
    }
    test_ds = test_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * output_shard_factor
    )
    train_ds = train_ds.flat_map(process_file, fn_kwargs=fn_kwargs).repartition(
        parallelism * output_shard_factor
    )

    splits = ("train", "test")

    for ds, split in zip((train_ds, test_ds), splits):
        ds.write_parquet(f"local://{os.path.abspath(output_dir)}/{split}")

    ray.shutdown()

    print(
        f"finished ray pipeline in {time.time() - start} secs. Files are written to {output_dir}"
    )

    for split in ("test", "train"):
        split_files = glob.glob(os.path.join(output_dir, split, "*.parquet"))
        print(f"converting {len(split_files)} files to webdataset for split {split}.")
        prefix = (
            split if not output_file_prefix else "-".join((output_file_prefix, split))
        )
        parquet_to_wds(
            split_files,
            prefix=prefix,
            chunk_size=chunk_size,
            output_dir=os.path.join(output_dir, split),
        )

    write_file_list(
        glob.glob(os.path.join(output_dir, "train", "train-*.tar")),
        os.path.join(output_dir, "train", f"train-files.txt"),
    )
    write_file_list(
        glob.glob(os.path.join(output_dir, "train", "traineval-*.tar")),
        os.path.join(output_dir, split, f"traineval-files.txt"),
    )
    write_file_list(
        glob.glob(os.path.join(output_dir, "test", "test-*.tar")),
        os.path.join(output_dir, "test", f"test-files.txt"),
    )
    return


if __name__ == "__main__":
    fire.Fire(main)