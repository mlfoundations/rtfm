from typing import Optional

from rtfm.task_config.configs import CONFIG_REGISTRY, TLMConfig


def get_tlm_config(task, override_config: Optional[str] = None) -> TLMConfig:
    """Get the TLMConfig for the given task."""
    # Set override_config to None *or* to an empty string to skip the override
    if override_config:
        return TLMConfig.from_yaml(override_config)

    try:
        return CONFIG_REGISTRY[task]
    except KeyError:
        raise KeyError(
            f"Task {task} missing from configs."
            f"Available configs are: {sorted(CONFIG_REGISTRY.keys())}"
        )
