import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

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
