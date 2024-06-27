from typing import Dict

import torch
from accelerate.utils import is_xpu_available


def batch_to_xpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if is_xpu_available():
        batch = {k: v.to("xpu") for k, v in batch.items()}
    elif torch.cuda.is_available():
        batch = {k: v.to("cuda") for k, v in batch.items()}
    return batch
