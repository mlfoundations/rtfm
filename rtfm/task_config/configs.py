import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Union, Dict, Optional, List, Any, Set

import yaml
from tableshift.core.features import FeatureList
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from rtfm.datasets.synthetic import SYNTHETIC_DATASET_NAMES
from rtfm.special_tokens import EOC_TOKEN

# USER_CONFIG_DIR is a colon-deliminted list of values that are searched for valid yaml files.
USER_CONFIG_DIRS = os.environ.get("USER_CONFIG_DIR", None)
if USER_CONFIG_DIRS:
    USER_CONFIG_DIRS = [os.path.abspath(d) + "/" for d in USER_CONFIG_DIRS.split(":")]
    USER_YAML_AND_DIR = [
        (f, dirpath)
        for dirpath in USER_CONFIG_DIRS
        for f in glob.glob(os.path.join(dirpath, "**", "*.yaml"), recursive=True)
    ]
    logging.info(f"found {len(USER_YAML_AND_DIR)} yaml files in {USER_CONFIG_DIRS}")
else:
    USER_YAML_AND_DIR = []

# Mapping of numeric targets to string values for binary classification
BINARY_CLS_LABELS_MAPPING = {"1": "Yes", "1.0": "Yes", "0": "No", "0.0": "No"}


@dataclass
class TLMConfig:
    """Class to hold the configs for a tabular LLM task."""

    prefix: str
    suffix: str
    task_context: Optional[str] = None

    # Optional mapping of labels to be applied.
    labels_mapping: Union[Dict[str, str], None] = field(
        default_factory=lambda: BINARY_CLS_LABELS_MAPPING
    )

    # List of the unique values that occur for the label. If none, the string
    # values in labels_mapping are used by default.
    label_values: Optional[List[Union[str, int]]] = None

    @property
    def string_label_values(self) -> Union[Set[str], None]:
        return set(str(x) for x in self.label_values) if self.label_values else None

    @property
    def reverse_labels_mapping(self) -> Dict[str, str]:
        return {v: k for v, k in self.labels_mapping}

    @classmethod
    def from_yaml(
        cls, yaml_file: str, required_fields=("prefix", "suffix", "labels_mapping")
    ):
        if not os.path.exists(yaml_file):
            raise ValueError(f"yaml file {yaml_file} does not exist.")
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        # Check the yaml file here to ensure the right fields are populated.
        for field_name in required_fields:
            if field_name not in config:
                logging.warning(
                    f"field {field_name} not in task config loaded from {yaml_file}"
                )

        return cls(**config)

    def to_yaml(self, filepath: str):
        with open(filepath, "w") as outfile:
            yaml.dump(self.__dict__, outfile, default_flow_style=False)
        return

    def map_label_value(self, y: Any) -> str:
        """Map from original --> encoded label names (i.e. 0 --> 'No')."""
        if self.labels_mapping is None:
            y_out = str(y)
            assert (
                y_out in self.string_label_values
            ), f"{y_out} not in {self.string_label_values}"
            return y_out
        try:
            return self.labels_mapping[y]
        except KeyError:
            raise KeyError(
                f" KeyError for key {y}; mapping is {self.labels_mapping}."
                f"Does the key type {type(y)} match the "
                f"expected type in mapping?"
            )

    def reverse_map_label_value(self, y: str) -> Any:
        """Map from encoded --> original label names (i.e. 'No'-->0."""
        return self.reverse_labels_mapping[y]

    def get_label_values(self) -> List[str]:
        """Get the list of possible labels for the task."""
        if self.string_label_values:
            return list(self.string_label_values)
        elif self.labels_mapping:
            return list(set(self.labels_mapping.values()))
        else:
            raise ValueError("no label values for task!")

    def get_prefix(self) -> str:
        """Get the full formatted prefix."""
        prefix = self.prefix.strip()
        if not prefix.endswith(":"):
            prefix += ":"
        return prefix

    def get_task_context(self) -> str:
        """Get the full formatted prefix."""
        ctx = self.task_context.strip() if self.task_context else ""
        return ctx

    def get_suffix(self):
        """Get the full formatted suffix."""
        suffix = self.suffix.strip()
        if not self.suffix.endswith("?"):
            suffix += "?"
        return suffix


def _mapped_target_values(f: FeatureList):
    tgt = f.target_feature
    if tgt.value_mapping is None:
        return None
    else:
        return list(tgt.value_mapping.values())


def make_config_registry() -> Dict[str, TLMConfig]:
    # For yaml files in the default config directory (CONFIGS_YAML_DIR),
    # we use the base name of the .yaml file as the task name.
    configs_dict = {}

    configs_dict.update(
        {
            k: TLMConfig(
                prefix="Predict the class label from the values of the features.",
                suffix="What is the class label?",
                task_context="This is an artificial dataset.",
                labels_mapping={
                    # Handle float labels by mapping to int; map e.g. '1.0' : '1'
                    **{str(float(x)): str(int(x)) for x in range(10)},
                    # Handle integer labels with identity mapping; map e.g. '1' : '1'
                    **{str(x): str(int(x)) for x in range(10)},
                },
            )
            for k in SYNTHETIC_DATASET_NAMES
        }
    )

    # For user-provided configs, we use the directory path, relative to
    # USER_CONFIG_DIR, as the task name.
    for yaml_path, dir_path in tqdm(USER_YAML_AND_DIR, desc="read user yaml files"):
        key = os.path.relpath(os.path.dirname(yaml_path), dir_path)
        try:
            configs_dict[key] = TLMConfig.from_yaml(yaml_path)
        except Exception as e:
            logging.error(
                f"exception reading yaml file: {e}; make sure the following is used "
                "to generate the yaml file to avoid styling issues:"
                """ 
        with open('data.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
                          """
            )

    return configs_dict


CONFIG_REGISTRY = make_config_registry()


def labels_max_len(task, tokenizer: PreTrainedTokenizer):
    """Compute the max length in tokens of all target classes (+ EOC token)."""
    max_len = -float("inf")
    longest_target = None
    for label in CONFIG_REGISTRY[task].get_label_values():
        label_text = label + EOC_TOKEN
        tokenized = tokenizer(label_text, return_attention_mask=False)
        tokens_len = len(tokenized["input_ids"]) + 1
        if tokens_len > max_len:
            max_len = tokens_len
            longest_target = label_text
    assert longest_target is not None, "sanity check that longest target is found."
    logging.info(f"got max len {max_len} for class {longest_target} in task " f"{task}")
    return max_len
