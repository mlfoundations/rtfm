"""
Configurations for non-Tableshift datasets.
"""
import copy
import glob
import logging
import os
from typing import Optional

import pandas as pd
from tableshift.configs.experiment_config import ExperimentConfig
from tableshift.configs.experiment_defaults import (
    DEFAULT_ID_VAL_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_ID_TEST_SIZE,
)
from tableshift.core import (
    RandomSplitter,
    PreprocessorConfig,
    DatasetConfig,
    TabularDataset,
)
from tableshift.core.data_source import (
    OfflineDataSource,
)
from tableshift.core.features import FeatureList
from tableshift.core.tabular_dataset import Dataset
from tableshift.core.tasks import TaskConfig, _TASK_REGISTRY as TABLESHIFT_TASK_REGISTRY

DEFAULT_SPLITTER = RandomSplitter(
    val_size=DEFAULT_ID_VAL_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    test_size=DEFAULT_ID_TEST_SIZE,
)


def make_default_config():
    """Construct a default ExperimentConfig for any experiments that do not require special configuration."""
    return ExperimentConfig(
        splitter=DEFAULT_SPLITTER,
        grouper=None,
        preprocessor_config=PreprocessorConfig(
            passthrough_columns="all",
        ),
        tabular_dataset_kwargs={},
    )


class AutoDataSource(OfflineDataSource):
    def __init__(
        self,
        name_or_path: str,
        preprocess_fn=lambda x: x,
        extension: str = "csv",
        **kwargs,
    ):
        self.name_or_path = name_or_path
        if not extension.startswith("."):
            extension = "." + extension
        self.extension = extension
        super().__init__(preprocess_fn=preprocess_fn, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        file_glob = os.path.join(
            self.cache_dir, self.name_or_path, "*" + self.extension
        )
        files = glob.glob(file_glob)
        file = files[0]
        if len(files) > 1:
            logging.warning(
                f"got {len(files)} files for data source {self.name_or_path},"
                f"taking the first one: {file}"
            )
        elif not len(files):
            raise ValueError(
                f"no files found matching {file_glob} for data source {self.name_or_path}"
            )
        if self.extension == ".csv":
            return pd.read_csv(file)
        else:
            raise ValueError(f"extension {self.extension} not yet implemented.")


class AutoConfig(TaskConfig):
    @classmethod
    def from_directory(cls, directory: str):
        assert os.path.exists(directory), f"{directory} does not exist."

        file_glob = os.path.join(directory, "*.jsonl")
        feature_list_jsonl = glob.glob(file_glob)
        if not len(feature_list_jsonl):
            raise FileNotFoundError(f"no feature list jsonl found matching {file_glob}")
        try:
            feature_list = FeatureList.from_jsonl(feature_list_jsonl[0])
        except Exception as e:
            # TODO: handle any exceptions here
            raise e

        return cls(AutoDataSource, feature_list)


def fetch_dataset_with_default_configs(
    name: str,
    cache_dir: str = "tmp",
    preprocessor_config: Optional[PreprocessorConfig] = None,
    initialize_data: bool = True,
    **kwargs,
) -> Dataset:
    """Get a dataset with the default configuration.

    This function is intended to "just get the data" for a given dataset name.
    It uses the default settings -- default task config, default experiment config,
    default dataset config -- to fetch the canonical version of a dataset
    with a specified name.

    This should be the default function for fetching a dataset, unless
    your intent is to somehow modify the task (in which case,
    you should call fetch_dataset_from_configs() directly).
    """
    # TaskConfig
    task_config = get_task_config(name, cache_dir=cache_dir)

    # ExperimentConfig
    expt_config = make_default_config()

    # DatasetConfig
    dataset_config = DatasetConfig(cache_dir=cache_dir)

    return fetch_dataset_from_configs(
        dataset_config=dataset_config,
        expt_config=expt_config,
        preprocessor_config=preprocessor_config,
        name=name,
        initialize_data=initialize_data,
        task_config=task_config,
        **kwargs,
    )


AMLB_TASKS = (
    "data_scientist_salary",
    "imdb_genre_prediction",
    "jigsaw_unintended_bias100K",
    "kick_starter_funding",
    "melbourne_airbnb",
    "news_channel",
    "product_sentiment_machine_hack",
    "wine_reviews",
)


def fetch_dataset_from_configs(
    name: str,
    task_config: TaskConfig,
    dataset_config: DatasetConfig,
    preprocessor_config: Optional[PreprocessorConfig] = None,
    expt_config: Optional[ExperimentConfig] = None,
    initialize_data=True,
    **kwargs,
) -> Dataset:
    """Fetches a tableshift.core.tabular_dataset.Dataset from the configs.

    This function should be used when the intent is to modify a task (e.g. by changing the
    FeatureList or task config); otherwise, it is recommended to use fetch_dataset_wit_default_configs()
    to fetch the canonical version of a dataset.
    """
    # Set up default ExperimentConfig and PreprocessorConfig, if not specified
    if expt_config is None:
        expt_config = make_default_config()

    if preprocessor_config is None:
        preprocessor_config = expt_config.preprocessor_config

    # Set up tabular dataset kwargs
    tabular_dataset_kwargs = copy.deepcopy(expt_config.tabular_dataset_kwargs)

    if "name" not in tabular_dataset_kwargs:
        tabular_dataset_kwargs["name"] = name

    if tabular_dataset_kwargs["name"] in AMLB_TASKS:
        tabular_dataset_kwargs.update(
            {"automl_benchmark_dataset_name": tabular_dataset_kwargs["name"]}
        )

    if isinstance(task_config, AutoConfig):
        tabular_dataset_kwargs["name_or_path"] = name

    # Fetch the dataset (either from cached or uncached)
    dset = TabularDataset(
        config=dataset_config,
        splitter=expt_config.splitter,
        grouper=kwargs.get("grouper", expt_config.grouper),
        preprocessor_config=preprocessor_config,
        initialize_data=initialize_data,
        task_config=task_config,
        **tabular_dataset_kwargs,
    )

    return dset


def get_task_config(task, cache_dir: str = None) -> TaskConfig:
    if task in TABLESHIFT_TASK_REGISTRY:
        task_config = TABLESHIFT_TASK_REGISTRY[task]
    else:
        logging.info(f"using AutoConfig for task {task}")
        task_config = AutoConfig.from_directory(os.path.join(cache_dir, task))
    return task_config
