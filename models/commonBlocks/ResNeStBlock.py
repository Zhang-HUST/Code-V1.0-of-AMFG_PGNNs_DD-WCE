import torch.nn as nn
import torch.nn.functional as F
from models.commonBlocks.ChannelAttentions import SplitAttBlock

"""
1. ResNeSt中的ResNeStConv和SplitAttBlock
2. SKBlock结构：ResNeStBottleneckBlock - Pooling - Dropout:
"""


class ResNeStConvBlock(nn.Module):
    """
    convolution block class

    convolution 2D -> batch normalization -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResNeStConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding='same', bias=False, ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResNeStBottleneckBlock(nn.Module):
    """
    bottleneck block class
    """
    expansion = 4

    def __init__(self, in_channels, channels, kernel_size, stride=1, dilation=(1, 1), downsample=None, radix=2,
                 groups=1, bottleneck_width=64, is_first=False):
        super(ResNeStBottleneckBlock, self).__init__()
        group_width = int(channels * (bottleneck_width / 64.)) * groups

        layers = [ResNeStConvBlock(in_channels=in_channels, out_channels=group_width, kernel_size=(1, 1), stride=(1, 1)),
                  SplitAttBlock(in_channels=group_width, channels=group_width, kernel_size=kernel_size,
                                stride=stride, padding='same', dilation=dilation, groups=groups, bias=False,
                                radix=radix)]

        if stride > 1 or is_first:
            layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=1))

        layers += [nn.Conv2d(group_width, channels * 4, kernel_size=(1, 1), padding='same', bias=False),
                   nn.BatchNorm2d(channels * 4)]

        self.block = nn.Sequential(*layers)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        if self.downsample:
            residual = self.downsample(x)
        out = self.block(x)
        out += residual

        return F.relu(out)


class ResNeStBlock(nn.Module):
    def __init__(self, resnestconv_dict, pooling_dict, dropout_dict, expansion=4):
        super(ResNeStBlock, self).__init__()
        self.resnestconv_dict = resnestconv_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict
        self.conv_layer = ResNeStConvBlock(in_channels=self.resnestconv_dict['in_channels'],
                                           out_channels=self.resnestconv_dict['out_channels'],
                                           kernel_size=self.resnestconv_dict['kernel'],
                                           stride=(1, 1))
        # ResNeStBottleneckBlock模块：
        self.resnestconv_layer = ResNeStBottleneckBlock(in_channels=self.resnestconv_dict['out_channels'],
                                                        channels=self.resnestconv_dict['out_channels'] // expansion,
                                                        kernel_size=self.resnestconv_dict['kernel'],
                                                        radix=self.resnestconv_dict['radix'],
                                                        groups=self.resnestconv_dict['groups']
                                                        )
        self.bn = nn.BatchNorm2d(self.resnestconv_dict['out_channels'])

        # activation 模块
        self.activation = nn.ReLU()
        # pool 模块：
        if self.pooling_dict['pooling_type'] == 'max2d':
            self.pool = nn.MaxPool2d(kernel_size=self.pooling_dict['pooling_kernel'],
                                     stride=self.pooling_dict['pooling_stride'])
        elif self.pooling_dict['pooling_type'] == 'ave2d':
            self.pool = nn.AvgPool2d(kernel_size=self.pooling_dict['pooling_kernel'],
                                     stride=self.pooling_dict['pooling_stride'])
        else:
            raise Exception(f"Not support pooling type {self.pooling_dict['pooling_type']} "
                            f"Only 'max2d' and 'ave2d' can be used, please check")
        # dropout 模块
        self.drop = nn.Dropout(p=self.dropout_dict['drop_rate'])

    def forward(self, x):
        y = self.conv_layer(x)
        # ResNeStBottleneckBlock：
        y = self.resnestconv_layer(y)
        y = self.bn(y)
        y = self.activation(y)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)

        return y
