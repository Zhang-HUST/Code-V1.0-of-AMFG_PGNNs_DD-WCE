import torch
import torch.nn as nn

"""
1. ResNetV1的基本残差连接结构：BasicResNetV1Block
2. ResNetV2中的基本残差连接结构：BasicResNetV2Block
3. ResNetBlock根据block_type来选择使用哪个ResNet版本，结构：ResNetBlock - Pooling - Dropout:
"""


# 定义基本的ResNetV1残差块
class BasicResNetV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicResNetV1Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)

        return out


# 定义基本的ResNetV2残差块
class BasicResNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicResNetV2Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding,
                               bias=False)
        self.downsample = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.downsample(identity)

        return out


# 定义完整的ResNetV1和ResNetV2残差块
class ResNetBlock(nn.Module):
    def __init__(self, resnet_dict, pooling_dict, dropout_dict, block_type):
        super(ResNetBlock, self).__init__()
        self.resnet_dict = resnet_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict

        # ResNet卷积模块：
        if block_type == 'ResNetV1':
            self.resnet_layer = BasicResNetV1Block(in_channels=self.resnet_dict['in_channels'],
                                                   out_channels=self.resnet_dict['out_channels'],
                                                   kernel_size=self.resnet_dict['kernel'],
                                                   stride=self.resnet_dict['stride'],
                                                   padding=self.resnet_dict['padding'],
                                                   )
        elif block_type == 'ResNetV2':
            self.resnet_layer = BasicResNetV2Block(in_channels=self.resnet_dict['in_channels'],
                                                   out_channels=self.resnet_dict['out_channels'],
                                                   kernel_size=self.resnet_dict['kernel'],
                                                   stride=self.resnet_dict['stride'],
                                                   padding=self.resnet_dict['padding'],
                                                   )
        else:
            raise ValueError('Unsupported ResNet block type: {}'.format(block_type))
        self.bn = nn.BatchNorm2d(self.resnet_dict['out_channels'])
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
        # ResNet卷积：
        y = self.resnet_layer(x)
        y = self.bn(y)
        y = self.activation(y)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)

        return y
