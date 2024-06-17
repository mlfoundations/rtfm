import glob
import os
from typing import Optional

import pandas as pd
from tableshift.core.features import FeatureList

from rtfm.data_sources.unipredict import format_target_column
from rtfm.task_config import TLMConfig


def clean_colname(colname: str) -> str:
    colname = colname.replace(".", "_")
    colname = colname.replace("__", "_")
    return colname


def get_task_csv_file(task_input_dir: str) -> str:
    """Fetch the CSV file from a task directory."""
    assert os.path.exists(
        task_input_dir
    ), f"Task input directory does not exist: {task_input_dir}"
    assert os.path.isdir(task_input_dir), f"expected directory {task_input_dir}"
    fileglob = os.path.join(task_input_dir, "*.csv")
    csv_files = glob.glob(fileglob)

    assert (
            len(csv_files) == 1
    ), f"expected one csv file matching {fileglob}, got {csv_files}"

    return csv_files[0]


def generate_files_from_csv(csv_src: str,
                            output_dir: str,
                            to_regression: bool,
                            task: Optional[str] = None):
    """Generate the FeatureList and TaskConfig for a CSV file,
    and write the results (including copy of CSV) to output_dir."""
    if task is None:
        task = os.path.basename(csv_src).replace(".csv", "")

    df = pd.read_csv(csv_src)

    df.columns = [clean_colname(colname) for colname in df.columns]

    target_colname = df.columns[-1]

    if to_regression:
        print(f"[INFO] converting target column {target_colname} to binned regression")
        df[target_colname] = format_target_column(df[target_colname])

    fl = FeatureList.from_dataframe(df, target_colname)

    os.makedirs(output_dir, exist_ok=True)

    feature_list_jsonl = os.path.join(output_dir, "feature_list.jsonl")
    fl.to_jsonl(feature_list_jsonl)
    print(f"[INFO] FeatureList written to {feature_list_jsonl}")

    task_config = TLMConfig(
        prefix=f"Predict the value of {target_colname}",
        suffix=f"What is the value of {target_colname}?",
        task_context=None,
        labels_mapping=None,
        label_values=df[target_colname].unique().tolist(),
    )

    task_config_yaml_path = os.path.join(output_dir, f"{task}.yaml")
    task_config.to_yaml(task_config_yaml_path)
    print(f"[INFO] TaskConfig written to {task_config_yaml_path}")

    csv_dest = os.path.join(output_dir, f"{task}.csv")
    df.to_csv(csv_dest, index=False)
    print(f"[INFO] CSV written to {csv_dest}")
    return
