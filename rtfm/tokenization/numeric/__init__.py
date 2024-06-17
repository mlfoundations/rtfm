from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn as nn


def cos_sin(x: Tensor) -> Tensor:
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


@dataclass
class PeriodicOptions:
    """Options for the periodic tokenizer."""

    n: int  # the output size is 2 * n
    sigma: float
    trainable: bool
    initialization: Literal["log-linear", "normal"]


class PeriodicTokenizerModule(nn.Module):
    """Adapted from https://github.com/yandex-research/tabular-dl-num-embeddings/blob/main/lib/deep.py"""

    def __init__(self, n_features: int, options: PeriodicOptions) -> None:
        super().__init__()
        if options.initialization == "log-linear":
            coefficients = options.sigma ** (torch.arange(options.n) / options.n)
            coefficients = coefficients[None].repeat(n_features, 1)
        else:
            assert options.initialization == "normal"
            coefficients = torch.normal(0.0, options.sigma, (n_features, options.n))
        if options.trainable:
            self.coefficients = nn.Parameter(coefficients)  # type: ignore[code]
        else:
            self.register_buffer("coefficients", coefficients)

    def forward(self, x: Tensor) -> Tensor:
        """Returns Tensor of shape [batch_size, num_features, 2 * options.n]."""
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients[None] * x[..., None])
