import torch
import torch.nn as nn

OPS = {
    '0': lambda C_in, C_out, stride, affine, track_running_stats, use_bn: Zero(C_in, C_out, stride),
    '1': lambda C_in, C_out, stride, affine, track_running_stats, use_bn: Identity() if (
                stride == 1 and C_in == C_out) else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats, use_bn),
    '2': lambda C_in, C_out, stride, affine, track_running_stats, use_bn: ConvBN(C_in, C_out, (1, 1), stride, (0, 0),
                                                                                 (1, 1), affine, track_running_stats, use_bn),
    '3': lambda C_in, C_out, stride, affine, track_running_stats, use_bn: ConvBN(C_in, C_out, (3, 3), stride, (1, 1),
                                                                                 (1, 1), affine, track_running_stats, use_bn)
}


class ConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, track_running_stats=True, use_bn=False):
        super(ConvBN, self).__init__()
        if use_bn:
            self.ops = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
            )
        else:
            self.ops = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, bias=False)
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        return self.ops(x)
    

class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, track_running_stats=True, use_bn=False):
        super(ConvBNReLU, self).__init__()
        if use_bn:
            self.ops = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
                nn.ReLU(inplace=False)
            )
        else:
            self.ops = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=False),
                nn.ReLU(inplace=False)
            )
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

    def forward(self, x):
        return self.ops(x)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1], shape[2], shape[3] = self.C_out, (shape[2] + 1) // self.stride, (shape[3] + 1) // self.stride
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros
        

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride, affine, track_running_stats, use_bn):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.use_bn = use_bn
        assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
        C_outs = [C_out // 2, C_out - C_out // 2]
        self.convs = nn.ModuleList()
        for i in range(2):
            self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        if self.use_bn:
            out = self.bn(out)
        return out


class DeconvLayer(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation=1,
                 affine=True, track_running_stats=True, use_bn=False):
        super(DeconvLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            C_in, C_out, kernel_size, stride, padding, output_padding=1)
        self.norm = None
        if use_bn:
            self.norm = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.relu(x)
        
        return x
