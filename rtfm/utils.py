import glob
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional

from rtfm.task_config.configs import CONFIG_REGISTRY


def get_experiments_list(
    experiments: List[str], exclude_experients: List[str]
) -> List[str]:
    """Fetch the list of task names, excluding as necessary."""
    if experiments == ["all"]:
        experiments = list(CONFIG_REGISTRY.keys())
    if exclude_experients:
        experiments = list(set(experiments) - set(exclude_experients))
    return experiments


def get_latest_checkpoint(dirpath) -> str:
    dirs = glob.glob(os.path.join(dirpath, "*step*"))
    assert len(dirs), f"no valid checkpoint directories found in {dirpath}"
    ckpt_dir = max(dirs, key=os.path.getmtime)
    return ckpt_dir


def initialize_dir(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(
            dirname, exist_ok=True
        )  # still set exist_ok=True to avoid race conditions


def timestamp() -> float:
    return datetime.now().timestamp()


def get_task_names_list(
    task_names: str, exclude_task_names: Optional[str] = None
) -> List[str]:
    """Fetch the list of task names, excluding as necessary."""
    if not task_names:
        return []

    task_names_in = task_names.replace("'", "")
    task_names_in = task_names_in.replace('"', "")

    task_names_out = []

    if task_names_in.endswith(".txt"):
        with open(task_names_in, "r") as f:
            for line in f:
                task_names_out.append(line.strip())
        return task_names_out

    task_names_in = task_names_in.split(",")
    assert len(task_names_in), "train_task_names_in list is empty!"
    assert isinstance(task_names_in, list)

    all_tasks = list(CONFIG_REGISTRY.keys())

    exclude_task_names = exclude_task_names.split(",") if exclude_task_names else None

    task_wildcard_chars = ["*"]

    for task_name in task_names_in:
        if any(char in task_name for char in task_wildcard_chars):
            # If task name is a wildcard; fetch any matching task names. Note that we add
            # the leading '^' character to only match on the beginning by default; this
            # prevents wildcards from returning tasks that contain another task as a substring
            # (i.e. "adult" could be in another task name).
            task_regex = re.compile("^" + task_name)
            task_names_out += [x for x in all_tasks if re.search(task_regex, x)]
        else:
            task_names_out.append(task_name)

    if exclude_task_names:
        task_names_out = list(set(task_names_out) - set(exclude_task_names))

    return list(set(task_names_out))


def write_text_file(text: str, filename: str) -> None:
    initialize_dir(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(text)


def write_json_file(elem: Dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(elem, f)
