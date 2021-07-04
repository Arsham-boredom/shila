"""QuartzNet, acoustic model for speech recognition 
based on paper: https://arxiv.org/abs/1910.10261
"""

from typing import List, Union
from dataclasses import dataclass

import torch
from torch.nn.modules import dropout, padding
from torch.utils import data

from src.nn import (
    TorchModule,
    DepthwiseSeperableConv1D
)


@dataclass
class PreConfig:
    input_channels: int
    kernel_size: int
    filter_size: int
    dropout: float


@dataclass
class BlockConfig:
    input_channels: List[int]
    filters: List[int]
    kernels: List[int]
    drop_rates: List[int]
    repeat: List[int] = 0

    def __str__(self) -> str:
        return f"neural config with {len(self.input_channels)} stack of convolution"


@dataclass
class PostConfig:
    input_channels: List[int]
    filters: List[int]
    kernels: List[int]
    drop_rates: List[int]


class PreBlock(TorchModule):
    def __init__(self, config: PreConfig) -> None:
        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.filter_size,
            kernel_size=config.kernel_size,
        )

        self.drop = torch.nn.Dropout(config.dropout)
        self.norm = torch.nn.BatchNorm1d(config.filter_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.drop(self.relu(self.norm(self.conv(x))))


def factory_gen(config: Union[BlockConfig, PostConfig]):

    if isinstance(config, PostConfig):
        return zip(
            config.input_channels,
            config.filters,
            config.kernels,
            config.drop_rates,
        )

    return zip(
        config.input_channels,
        config.filters,
        config.kernels,
        config.drop_rates,
        config.repeat
    )


class PostBlock(TorchModule):
    def __init__(self, config: PostConfig) -> None:
        super().__init__()

        self.model = torch.nn.Sequential()

        for idx, cell_config in enumerate(factory_gen(config), start=1):
            # extract parameters for each single cell
            in_channels, out_channels, kernel, drop = cell_config

            self.model.add_module(
                f"Conv1d::{idx}",
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                )
            )

            self.model.add_module(f"Drop::{idx}", torch.nn.Dropout(drop))
            self.model.add_module(
                f"BatchNorm::{idx}",
                torch.nn.BatchNorm1d(out_channels)
            )
            self.model.add_module(
                f"RELU::{idx}",
                torch.nn.ReLU()
            )

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

        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class TimeChannel(TorchModule):
    def __init__(self, in_channels, out_channels, kernel_size, drop) -> None:
        super().__init__()

        self.conv = DepthwiseSeperableConv1D(
            input_channels=in_channels,
            output_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            dropout=drop
        )

        self.point_wise = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.point_wise(self.conv(x))
        return self.relu(self.norm(x))


class QuartzSubBlock(TorchModule):
    def __init__(self, in_channels, out_channels, kernel_size, drop, repeat) -> None:
        super().__init__()

        self.pipe = torch.nn.Sequential()

        self.pipe.add_module(
            "TimeChannel::first",
            TimeChannel(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                drop=drop
            )
        )

        for idx in range(repeat - 1):
            self.pipe.add_module(
                "TimeChannel::{}".format(idx),
                TimeChannel(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    drop=drop
                )
            )

        self.last_sub_block = torch.nn.Sequential(
            DepthwiseSeperableConv1D(
                input_channels=out_channels,
                output_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2
            ),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm1d(out_channels)
        )

        self.residual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
            torch.nn.BatchNorm1d(out_channels)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.pipe(x)
        residual = self.residual(x)
        output = self.last_sub_block(output) + residual
        return self.relu(output)


class QuartzBlock(TorchModule):
    def __init__(self, config: BlockConfig) -> None:
        super().__init__()

        self.model = torch.nn.Sequential()

        for idx, cell_config in enumerate(factory_gen(config), start=1):
            in_channels, out_channels, kernel, drop, repeat = cell_config

            self.model.add_module(
                "Sub-Block::{}".format(idx),
                QuartzSubBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    drop=drop,
                    repeat=repeat
                )
            )

    def forward(self, x):
        return self.model(x)


class QuartzNet(TorchModule):
    def __init__(
        self,
        pre_config: PreConfig,
        block_config: BlockConfig,
        post_config: PostConfig,
    ):

        super().__init__()

        self.pre = PreBlock(pre_config)
        self.blocks = QuartzBlock(block_config)
        self.post = PostBlock(post_config)

    def forward(self, x):
        return self.post(self.blocks(self.pre(x)))
