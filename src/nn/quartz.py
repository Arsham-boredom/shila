"""QuartzNet, acoustic model for speech recognition 
based on paper: https://arxiv.org/abs/1910.10261
"""

from typing import List
from dataclasses import dataclass

import torch

from src.nn import (
    TorchModule,
    DepthwiseSeperableConv1D
)


@dataclass
class PreConfig:
    kernel_size: int
    filter_size: int
    drouput: float


@dataclass
class BlockConfig:
    filters: List[int]
    kernels: List[int]
    drop_rates: List[int]


class PostConfig(BlockConfig):
    dilation_rates: List[int]


class PreBlock(TorchModule):
    def __init__(self, input_channels, kernel_size, filter_size, dropout) -> None:
        super().__init__()

        self.conv = DepthwiseSeperableConv1D(
            input_channels=input_channels,
            output_channels=filter_size,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.norm = torch.nn.BatchNorm1d()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class QuartzBlock(TorchModule):
    pass

class QuartzNet(TorchModule):
    def __init__(
        self,
        pre_config: PreConfig,
        block_config: BlockConfig,
        post_config: PostConfig,
    ):

        super().__init__()
