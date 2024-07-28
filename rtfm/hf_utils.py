import os
from typing import Union

from llama_recipes.model_checkpointing.checkpoint_handler import fullstate_save_policy
from rtfm.configs import TrainConfig
from torch.distributed import fsdp as FSDP
from torch.distributed.fsdp import StateDictType


def fetch_auth_token() -> Union[str, None]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]


def save_hf_model_and_tokenizer(model, tokenizer, train_config: TrainConfig):
    """Save model and tokeninzer in hugging face format for easy loading."""
    save_directory = train_config.make_save_folder_name()
    if train_config.enable_fsdp:
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            cpu_state = model.state_dict()

            print(f"saving process: rank {rank}  done w model state_dict\n")

        model.save_pretrained(state_dict=cpu_state, save_directory=save_directory)
    else:
        model.save_pretrained(save_directory=save_directory)

    tokenizer.save_pretrained(save_directory=save_directory)

    return
