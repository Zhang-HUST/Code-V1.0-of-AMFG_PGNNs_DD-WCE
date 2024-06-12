import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. ShuffleNetV1：ShuffleNetV1UnitA (stride=1，通道不升维)，ShuffleNetV1UnitB (原始stride=2，通道升维，这里改为stride=1)
   ShuffleNetV1Block：[ShuffleNetV1UnitA, ShuffleNetV1UnitB] - Pooling - Dropout.
   
1. ShuffleNetV2：ShuffleNetUnitA (stride=1，通道不升维)，ShuffleNetUnitB (原始stride=2，通道升维，这里改为stride=1)
   ShuffleNetBlock：[ShuffleNetUnitA, ShuffleNetUnitB] - Pooling - Dropout.
"""


# 通用的通道打乱模块
def channel_shuffle(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    out = x.view(batch_size, channels, height, width)

    return out


"""ShuffleNetV1"""


# ShuffleNetV1的stride=1模块，通道不升维
class ShuffleNetV1UnitA(nn.Module):
    """ShuffleNet unit for stride=1"""

    def __init__(self, in_channels, out_channels, kernel, groups):
        super(ShuffleNetV1UnitA, self).__init__()
        assert in_channels == out_channels
        # assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels, (1, 1), groups=groups, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel, padding='same', stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels, (1, 1), stride=1, padding='same', groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = channel_shuffle(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)

        return out


# ShuffleNetV1的原stride=2（这里改为stride=1）模块，通道升维
class ShuffleNetV1UnitB(nn.Module):
    """ShuffleNet unit for stride=2"""

    def __init__(self, in_channels, out_channels, kernel, groups, branch):
        super(ShuffleNetV1UnitB, self).__init__()
        self.kernel = kernel
        # assert out_channels % 4 == 0
        out_channels -= in_channels
        if branch == 1:
            bottleneck_channels = out_channels // 4
        else:
            bottleneck_channels = int(out_channels // 2)

        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels, (1, 1), groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel, padding='same', stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels, (1, 1), stride=1, groups=groups)  # stride=2
        self.bn6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = channel_shuffle(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, self.kernel, stride=1, padding=self.judge_padSize(self.kernel))
        # print(x.size(), out.size())
        out = F.relu(torch.cat([x, out], dim=1))
        # print(out.size())
        return out

    def judge_padSize(self, k):
        if int(k[0]) == 1 and int(k[1]) == 1:
            padding = (0, 0)
        else:
            kernel = int(k[0]) if int(k[0]) > 1 else int(k[1])
            padSize = int((kernel - 1) / 2)
            temp = [0, 0]
            for i, x in enumerate(k):
                if x > 1:
                    temp[i] = padSize
            padding = tuple(temp)

        return padding


"""ShuffleNetV2"""


def conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding='same', bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=(1, 1), stride=1, padding='same', bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )


class ShuffleNetV2InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel, stride, benchmodel):
        super(ShuffleNetV2InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, kernel_size=(1, 1), stride=1, padding='same', bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(),
                # dw
                nn.Conv2d(oup_inc, oup_inc, kernel, stride, padding='same', groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, kernel_size=(1, 1), stride=1, padding='same', bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel, stride, padding='same', groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, kernel_size=(1, 1), stride=1, padding='same', bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, kernel_size=(1, 1), stride=1, padding='same', bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, kernel, stride, padding='same', groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, kernel_size=(1, 1), stride=1, padding='same', bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        global out
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


# 定义完整的ShuffleNetV1和ShuffleNetV2残差块
class ShuffleNetBlock(nn.Module):
    def __init__(self, shufflenet_dict, pooling_dict, dropout_dict, block_type, branch=1):
        super(ShuffleNetBlock, self).__init__()
        self.shufflenet_dict = shufflenet_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict

        # ShuffleNet卷积模块：
        if block_type == 'ShuffleNetV1':
            self.shufflenet_layer = nn.Sequential(ShuffleNetV1UnitB(in_channels=self.shufflenet_dict['in_channels'],
                                                                    out_channels=self.shufflenet_dict['out_channels'],
                                                                    kernel=self.shufflenet_dict['kernel'],
                                                                    groups=self.shufflenet_dict['groups'],
                                                                    branch=branch),
                                                  ShuffleNetV1UnitA(in_channels=self.shufflenet_dict['out_channels'],
                                                                    out_channels=self.shufflenet_dict['out_channels'],
                                                                    kernel=self.shufflenet_dict['kernel'],
                                                                    groups=self.shufflenet_dict['groups']), )

        elif block_type == 'ShuffleNetV2':
            self.shufflenet_layer = nn.Sequential(ShuffleNetV2InvertedResidual(inp=self.shufflenet_dict['in_channels'],
                                                                               oup=self.shufflenet_dict['out_channels'],
                                                                               kernel=self.shufflenet_dict['kernel'],
                                                                               stride=1,
                                                                               benchmodel=2),
                                                  ShuffleNetV2InvertedResidual(inp=self.shufflenet_dict['out_channels'],
                                                                               oup=self.shufflenet_dict['out_channels'],
                                                                               kernel=self.shufflenet_dict['kernel'],
                                                                               stride=1,
                                                                               benchmodel=1), )
        else:
            raise ValueError('Unsupported ResNet block type: {}'.format(block_type))
        self.bn = nn.BatchNorm2d(self.shufflenet_dict['out_channels'])
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
        # ShuffleNet卷积：
        y = self.shufflenet_layer(x)
        y = self.bn(y)
        y = self.activation(y)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)

        return y
