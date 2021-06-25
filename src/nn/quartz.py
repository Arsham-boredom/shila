"""QuartzNet, acoustic model for speech recognition 
based on paper: https://arxiv.org/abs/1910.10261
"""

from typing import List, Union
from dataclasses import dataclass

import torch
from torch.nn.modules import dropout

from src.nn import (
    TorchModule,
    DepthwiseSeperableConv1D
)


@dataclass
class PreConfig:
    input_channels: int
    kernel_size: int
    filter_size: int
    drouput: float


@dataclass
class BlockConfig:
    input_channels: List[int]
    filters: List[int]
    kernels: List[int]
    drop_rates: List[int]


class PostConfig(BlockConfig):
    dilation_rates: List[int]


class PreBlock(TorchModule):
    def __init__(self, config: PreConfig) -> None:
        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.filter_size,
            kernel_size=config.kernel_size,
        )

        self.drop = torch.nn.Dropout()
        self.norm = torch.nn.BatchNorm1d()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.drop(self.relu(self.norm(self.conv(x))))


def factory_gen(config: Union[BlockConfig, PostConfig]):
    return zip(
        config.input_channels,
        config.filters,
        config.kernels,
        config.drop_rates,
        None if isinstance(config, BlockConfig) else config.dilation_rates
    )


class PostBlock(TorchModule):
    def __init__(self, config: PostConfig) -> None:
        super().__init__()

        self.model = torch.nn.Sequential()

        for cell_config in factory_gen(config):
            # extract parameters for each single cell
            in_channels, out_channels, kernel, drop, _ = cell_config

            self.model.add_module(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                )
            )

            self.model.add_module(torch.nn.BatchNorm1d())
            self.model.add_module(torch.nn.ReLU())
            self.model.add_module(torch.nn.Dropout())

        self.pointwise = torch.nn.Conv1d(
            in_channels=config.filters[-1],
            out_channels=config.filters[-1],
            kernel_size=1
        )

    def forward(self, x):
        return self.pointwise(self.model(x))

class QuartzSubBlock(TorchModule):
    def __init__(self, in_channels, out_channels, kernel_size, drop) -> None:
        super().__init__()

        self.conv = DepthwiseSeperableConv1D(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=kernel_size,
            dropout=drop
        )

        self.norm = torch.nn.BatchNorm1d()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))



class QuartzBlock(TorchModule):
    def __init__(self, config: BlockConfig) -> None:
        super().__init__()

        self.pipe = torch.nn.Sequential()

        block_length = len(config.input_channels)

        for idx, cell_config in enumerate(factory_gen(config), start=1):
            in_channels, out_channels, kernel, drop, _ = cell_config

            self.pipe.add_module(
                DepthwiseSeperableConv1D(
                    input_channels=in_channels,
                    output_channels=out_channels,
                    kernel_size=kernel,
                    dropout=drop
                )
            )

            self.pipe.add_module(torch.nn.BatchNorm1d())
            self.pipe.add_module(torch.nn.ReLU())

            if idx == (block_length -1):
                break

        # last last sub-block without relu activation
        self.pipe.add_module(
            DepthwiseSeperableConv1D(
                input_channels=config.input_channels[-1],
                output_channels=config.filters[-1],
                kernel_size=config.kernels[-1],
                dropout=config.drop_rates[-1]
            )
        )

        self.pipe.add_module(torch.nn.BatchNorm1d())

        self.residual = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=config.input_channels[0],
                out_channels=config.filters[-1],
                kernel_size=1
            ),

            torch.nn.BatchNorm1d()
        )

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pipe(x)
        residual = self.residual(x)
        return self.relu(x + residual)


class QuartzNet(TorchModule):
    def __init__(
        self,
        pre_config: PreConfig,
        block_config: BlockConfig,
        post_config: PostConfig,
    ):

        super().__init__()

        self.pre = PreBlock(pre_config)
        self.blocks = BlockConfig(block_config)
        self.post = PostBlock(post_config)

    def forward(self, x):
        return self.post(self.blocks(self.pre(x)))