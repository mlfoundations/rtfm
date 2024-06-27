from typing import Dict, Any

import tableshift.core
from tableshift import get_iid_dataset
from tableshift.exceptions import ConfigNotFoundException

from rtfm.datasets.configs import AutoConfig, fetch_dataset_with_default_configs
from rtfm.datasets.configs import (
    make_default_config,
)


def get_task_dataset(
    task,
    preprocessor_config=None,
    initialize_data=True,
    tabular_dataset_kwargs: Dict[str, Any] = None,
) -> tableshift.core.tabular_dataset.Dataset:
    if tabular_dataset_kwargs is None:
        tabular_dataset_kwargs = {}
    try:
        tabular_dataset = get_iid_dataset(
            task,
            cache_dir="tmp",
            use_cached=False,
            preprocessor_config=preprocessor_config,
            grouper=None,
            initialize_data=initialize_data,
            **tabular_dataset_kwargs,
        )
    except ConfigNotFoundException:
        # tabular_dataset_kwargs should not be specified
        # when using default configs.
        assert not tabular_dataset_kwargs

        tabular_dataset = fetch_dataset_with_default_configs(
            task,
            preprocessor_config=preprocessor_config,
            grouper=None,
            initialize_data=initialize_data,
        )
    assert tabular_dataset is not None, f"no dataset found for {task}"
    return tabular_dataset
