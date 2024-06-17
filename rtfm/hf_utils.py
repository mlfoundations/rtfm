import os
from typing import Union


def fetch_auth_token() -> Union[str, None]:
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
