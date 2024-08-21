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
