from typing import Text
from torch import nn
from src.nn import TorchModule

class DynamicDepthwiseSeperableConv(TorchModule):
    """Higher order class that implement depthwise seperable convolution 
    in 1D and 2D
    """

    __Conv1d__ = '1d'
    __Conv2d__ = '2d'

    def __init__(self, dimention: Text, input_channels, output_channels, kernel_size=1, stride=1, dropout=None):
        super().__init__()

        if dimention == self.__Conv1d__:
            conv = nn.Conv1d
        elif dimention == self.__Conv2d__:
            conv = nn.Conv2d

        self.depthwise = conv(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            stride=stride
        )

        self.pointwise = conv(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=stride
        )

        self.model = nn.Sequential(
            self.depthwise,
            self.pointwise
        )

        self.dropout = nn.Dropout(dropout)
        if dropout is not None:
            self.model.add_module("dropout", self.dropout)

    def forward(self, x):
        return self.model(x)


class DepthwiseSeperableConv2d(DynamicDepthwiseSeperableConv):
    """ 2D depthwise seperable convolution
    """

    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, dropout=None):
        super().__init__('2d', input_channels, output_channels, kernel_size=kernel_size, stride=stride, dropout=dropout)


class DepthwiseSeperableConv1D(DynamicDepthwiseSeperableConv):
    """ 1D depthwise seperable convolution
    """

    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, dropout=None):
        super().__init__('1d', input_channels, output_channels, kernel_size=kernel_size, stride=stride, dropout=dropout)

