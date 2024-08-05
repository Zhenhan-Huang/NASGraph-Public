import torch

import torch.nn as nn

OPS = {
    'none': lambda C, stride, affine, use_bn: Zero(stride),
    'avg_pool_2x2': lambda C, stride, affine, use_bn: nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
    'avg_pool_3x3': lambda C, stride, affine, use_bn: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5x5': lambda C, stride, affine, use_bn: nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
    'max_pool_2x2': lambda C, stride, affine, use_bn: nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda C, stride, affine, use_bn: nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda C, stride, affine, use_bn: nn.MaxPool2d(5, stride=stride, padding=2),
    'max_pool_7x7': lambda C, stride, affine, use_bn: nn.MaxPool2d(7, stride=stride, padding=3),
    'skip_connect': lambda C, stride, affine, use_bn: Identity() if stride == 1 else FactorizedReduce(C, C, use_bn=use_bn, affine=affine),
    'conv_1x1': lambda C, stride, affine, use_bn: ConvBN(C, C, 1, stride, 0, use_bn=use_bn, affine=affine),
    'conv_3x3': lambda C, stride, affine, use_bn: ConvBN(C, C, 3, stride, 1, use_bn=use_bn, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, use_bn: SepConv(C, C, 3, stride, 1, use_bn=use_bn, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine, use_bn: SepConv(C, C, 5, stride, 2, use_bn=use_bn, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine, use_bn: SepConv(C, C, 7, stride, 3, use_bn=use_bn, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine, use_bn: DilConv(C, C, 3, stride, 2, 2, use_bn=use_bn, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine, use_bn: DilConv(C, C, 5, stride, 4, 2, use_bn=use_bn, affine=affine),
    'dil_sep_conv_3x3': lambda C, stride, affine, use_bn: DilSepConv(C, C, 3, stride, 2, 2, use_bn=use_bn, affine=affine),
    'conv_3x1_1x3': lambda C, stride, affine, use_bn: ConvConvBN(C, C, 3, stride, 1, use_bn=use_bn, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine, use_bn: ConvConvBN(C, C, 7, stride, 3, use_bn=use_bn, affine=affine)
}


class NDS_CIFAR_HEAD(nn.Module):
    def __init__(self, C_in, C_out, use_bn=False):
        super().__init__()
        if use_bn:
            self.stem = nn.Sequential(
                nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=False)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
                nn.ReLU(inplace=False)
            )
    
    def forward(self, x):
        return self.stem(x)


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, use_bn=False, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        y = self.pad(x)
        out = torch.cat([self.conv1(x), self.conv2(y[:, :, 1:, 1:])], dim=1)
        if self.use_bn:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, use_bn=False, affine=False):
        super().__init__()
        if use_bn:
            self.module = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel, stride=stride, padding=padding, bias=False),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.module(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, use_bn=False, affine=False):
        super().__init__()
        if use_bn:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )

        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation, use_bn=False, affine=False):
        super().__init__()
        if use_bn:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )
        
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.op(x)


class DilSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation, use_bn=False, affine=False):
        super().__init__()
        if use_bn:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=1, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=kernel, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )

        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel, stride=1, padding=padding, dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=kernel, padding=0, bias=False),
                nn.ReLU(inplace=False)
            )
    def forward(self, x):
        return self.op(x)
    

class ConvConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, use_bn=False, affine=False):
        super().__init__()
        if use_bn:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, (1, kernel), stride=(1, stride), padding=(0, 1), bias=False),
                nn.Conv2d(C_in, C_out, (kernel, 1), stride=(stride, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=False)
            )
        
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, (1, kernel), stride=(1, stride), padding=(0, 1), bias=False),
                nn.Conv2d(C_in, C_out, (kernel, 1), stride=(stride, 1), padding=(1, 0), bias=False),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.op(x)
