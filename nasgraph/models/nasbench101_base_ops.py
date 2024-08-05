import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_bn=True):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3x3BnRelu(nn.Module):
    """3x3 convolution with batch norm and ReLU activation."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv3x3BnRelu, self).__init__()

        self.conv3x3 = ConvBnRelu(in_channels, out_channels, 3, 1, 1, **kwargs)

    def forward(self, x):
        x = self.conv3x3(x)
        return x


class Conv1x1BnRelu(nn.Module):
    """1x1 convolution with batch norm and ReLU activation."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1x1BnRelu, self).__init__()

        self.conv1x1 = ConvBnRelu(in_channels, out_channels, 1, 1, 0, **kwargs)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class MaxPool3x3(nn.Module):
    """3x3 max pool with no subsampling."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs):
        super(MaxPool3x3, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x


# Commas should not be used in op names
OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}
