import torch
import torch.nn as nn
from torch.nn import init
from models.commonBlocks.ChannelAttentions import SEAttBlock, SKAttBlock, ECAAttBlock, SplitAttBlock
from models.commonBlocks.SpatialAttentions import SelfAttBlock, NonLocalAttBlock, CCAttBlock, GCAttBlock
from models.commonBlocks.ChannelSpatialAttentions import BAMAttBlock, CBAMAttBlock, DAAttBlock, DAHeadAttBlock
from models.commonBlocks.ChannelSpatialAttentions import CAAttBlock, SAAttBlock, TripletAttBlock, SCAttBlock
from models.commonBlocks.ActivationFunctions import HardSwish

"""
1. MobileNetV1的深度可分离卷积：DepthWiseSeparableConv
2. MobileNetV2中的残差块结构：InvertedResidual
3. MobileNetV3中的瓶颈块结构：BottleNeckBlock，包括HardSwish和HardSigmoid激活、注意力机制（默认为SEBlock，可自行替换）
4. MobileNetBlock根据block_type来选择使用哪个MobileNet版本，结构：
    1）MobileNetV1：DSConv - BN - Activation - Pooling - Dropout;
    2）MobileNetV2/V3：InvertedResidual/BottleNeckBlock - Pooling - Dropout.
"""


# 定义MobileNetV1的深度可分离卷积
class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthWiseSeparableConv, self).__init__()
        self.ch_in = in_channels
        self.ch_out = out_channels
        self.depth_conv = nn.Conv2d(self.ch_in, self.ch_in, kernel_size=kernel_size, padding='same', groups=self.ch_in)
        self.point_conv = nn.Conv2d(self.ch_in, self.ch_out, kernel_size=(1, 1))

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# MobileNetV2中的DW卷积
def Conv3x3BNReLU(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        # stride=2 wh减半，stride=1 wh不变
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding='same', groups=in_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )


# MobileNetV2中的PW卷积
def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                  padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6()
    )


# MobileNetV2中的PW卷积(Linear) 没有使用激活函数
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                  padding='same'),
        nn.BatchNorm2d(out_channels)
    )


# MobileNetV2中的关键块：倒残差块
class InvertedResidual(nn.Module):
    # t = expansion_factor,也就是扩展因子，文章中取6
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        mid_channels = (in_channels * expansion_factor)

        # 先1x1卷积升维，再1x1卷积降维
        self.bottleneck = nn.Sequential(
            # 升维操作: 扩充维度是 in_channels * expansion_factor (6倍)
            Conv1x1BNReLU(in_channels, mid_channels),
            # DW卷积,降低参数量
            Conv3x3BNReLU(mid_channels, mid_channels, kernel_size, stride),
            # 降维操作: 降维度 in_channels * expansion_factor(6倍) 降维到指定 out_channels 维度
            Conv1x1BN(mid_channels, out_channels)
        )

        # 第一种: stride=1 才有shortcut 此方法让原本不相同的channels相同
        # if self.stride == (1, 1):
        #     self.shortcut = Conv1x1BN(in_channels, out_channels)
        # 第二种: stride=1: in_channels != out_channels时经过一次卷积，in_channels = out_channels时self.shortcut = None，
        self.shortcut = nn.Sequential()
        if stride == (1, 1) and in_channels != out_channels:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        # 第一种:
        # out = (out + self.shortcut(x)) if self.stride == (1, 1) else out
        # 第二种:
        out = (out + self.shortcut(x)) if self.stride == (1, 1) else out
        # out = (out + x) if self.stride == (1, 1) and self.in_channels == self.out_channels else out
        return out


# MobileNetV3中线性瓶颈和反向残差结构: BottleNeckBlock
class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, expand_size, out_channels, kernel_size, stride, nonlinear, attention):
        super(BottleNeckBlock, self).__init__()
        self.stride = stride
        self.attention = attention
        if self.attention == 'None':
            # self.attention_layer = nn.Sequential()
            pass
        # 通道注意力
        elif self.attention == 'SE':
            # activation: one of ['sigmoid', 'hard-sigmoid']
            self.attention_layer = SEAttBlock(out_channels, activation='hard-sigmoid')
        elif self.attention == 'SK':
            kernel_size_sk = judge_kernel_size_in_sk(kernel_size)
            self.attention_layer = SKAttBlock(out_channels, kernel_list=kernel_size_sk, G=16, r=8, stride=(1, 1), L=32)
        elif self.attention == 'ECA':
            self.attention_layer = ECAAttBlock(channel=out_channels, gamma=2, b=1)
        elif self.attention == 'Split':
            self.attention_layer = SplitAttBlock(in_channels=out_channels, channels=out_channels,
                                                 kernel_size=kernel_size, groups=1, radix=2, reduction_factor=4)
        # 空间注意力，类似于NLP的自注意力
        elif self.attention == 'SelfAtt':
            self.attention_layer = SelfAttBlock(in_dim=out_channels)
        elif self.attention == 'NonLocal':
            self.attention_layer = NonLocalAttBlock(channel=out_channels)
        elif self.attention == 'CC':
            self.attention_layer = CCAttBlock(in_dim=out_channels, reduction=8)
        elif self.attention == 'GC':
            self.attention_layer = GCAttBlock(inplanes=out_channels, ratio=0.25)
        # 混合的通道和空间注意力
        elif self.attention == 'BAM':
            kernel_size_bam = judge_kernel_size_in_bam_cru(kernel_size)
            self.attention_layer = BAMAttBlock(out_channels, kernel_size=kernel_size_bam, reduction_ratio=16,
                                               dilation=(1, 1))
        elif self.attention == 'CBAM':
            kernel_size_cbam = judge_kernel_size_in_cbam(kernel_size)
            self.attention_layer = CBAMAttBlock(out_channels, kernel_size=kernel_size_cbam, reduction_ratio=16,
                                                dilation=(1, 1))
        elif self.attention == 'DA':
            # self.attention_layer = DAAttBlock(in_channels=out_channels, reduction=8)
            self.attention_layer = DAHeadAttBlock(in_channels=out_channels, kernel_size=kernel_size, reduction=8)
        elif self.attention == 'CA':
            self.attention_layer = CAAttBlock(inp=out_channels, oup=out_channels, reduction=32)
        elif self.attention == 'SA':
            # out_channels: 64, 96, 128, 256, group = 8, 12, 16, 32
            group = out_channels // 8
            self.attention_layer = SAAttBlock(channel=out_channels, groups=group)
        elif self.attention == 'Triplet':
            self.attention_layer = TripletAttBlock(kernel_size=kernel_size)
        elif self.attention == 'SC':
            kernel_size_cru = judge_kernel_size_in_bam_cru(kernel_size)
            self.attention_layer = SCAttBlock(op_channel=out_channels, group_kernel_size=kernel_size_cru)
        else:
            raise ValueError('attention should be one of [SE, SK, ECA, Split, SelfAtt, NonLocal, CC, GC, BAM, '
                             'CBAM, DA,  CA, SA, Triplet, SC]')

        if nonlinear == 'ReLU':
            self.nonlinear1, self.nonlinear2 = nn.ReLU(), nn.ReLU()
        elif nonlinear == 'HardSwish':
            self.nonlinear1, self.nonlinear2 = HardSwish(), HardSwish()
        else:
            raise ValueError('nonlinear should be ReLU or HardSwish')

        # 1*1展开卷积
        self.conv1 = nn.Conv2d(in_channels, expand_size, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        # 3*3（或5*5）深度可分离卷积
        # padding=kernel_size // 2
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding='same', groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        # 1*1投影卷积
        self.conv3 = nn.Conv2d(expand_size, out_channels, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == (1, 1) and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding='same', bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.nonlinear1(self.bn1(self.conv1(x)))
        out = self.nonlinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 注意力模块
        if self.attention == 'None':
            pass
        else:
            self.attention_layer(out)
        # 残差链接
        out = out + self.shortcut(x) if self.stride == (1, 1) else out

        return out


# 定义完整的MobileNetV1, MobileNetV2和MobileNetV3块
class MobileNetBlock(nn.Module):
    def __init__(self, mobilenet_dict, bn_dict, pooling_dict, dropout_dict, block_type):
        super(MobileNetBlock, self).__init__()
        self.mobilenet_dict = mobilenet_dict
        self.bn_dict = bn_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict
        self.block_type = block_type

        # MobileNet卷积模块：
        if self.block_type == 'MobileNetV1':
            self.mobilenet_layer = DepthWiseSeparableConv(in_channels=self.mobilenet_dict['in_channels'],
                                                          out_channels=self.mobilenet_dict['out_channels'],
                                                          kernel_size=self.mobilenet_dict['kernel_size'])
        elif self.block_type == 'MobileNetV2':
            self.mobilenet_layer = self.make_mobilenetv2_layer(in_channels=self.mobilenet_dict['in_channels'],
                                                               out_channels=self.mobilenet_dict['out_channels'],
                                                               kernel_size=self.mobilenet_dict['kernel_size'],
                                                               stride=self.mobilenet_dict['stride'],
                                                               factor=self.mobilenet_dict['factor'],
                                                               block_num=self.mobilenet_dict['block_num'])
        elif self.block_type == 'MobileNetV3':
            self.mobilenet_layer = BottleNeckBlock(in_channels=self.mobilenet_dict['in_channels'],
                                                   expand_size=self.mobilenet_dict['expand_size'],
                                                   out_channels=self.mobilenet_dict['out_channels'],
                                                   kernel_size=self.mobilenet_dict['kernel_size'],
                                                   stride=self.mobilenet_dict['stride'],
                                                   nonlinear=self.mobilenet_dict['nonlinear'],
                                                   attention=self.mobilenet_dict['attention'])
        else:
            raise ValueError('Unsupported MobileNet block type: {}'.format(self.block_type))

        # bacth normalization 模块
        self.bn = nn.BatchNorm2d(self.mobilenet_dict['out_channels'])

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

    def make_mobilenetv2_layer(self, in_channels, out_channels, kernel_size, stride, factor, block_num):
        layers = [InvertedResidual(in_channels, out_channels, kernel_size, stride, factor)]
        # 这些叠加层stride均为1，in_channels = out_channels, 其中 block_num-1 为重复次数
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, kernel_size, (1, 1), factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        # MobileNet卷积模块：
        y = self.mobilenet_layer(x)
        # BN:
        if self.bn_dict['use_BN']:
            y = self.bn(y)  # BN
        # Activation:
        y = self.activation(y)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)

        return y


# 若BottleNeckBlock中使用BAM注意力机制，返回(1, 3)或(3, 1)的kernel_size
# 对SCBlock中的CRU注意力机制，返回(1, 3)或(3, 1)的kernel_size
def judge_kernel_size_in_bam_cru(k):
    if int(k[0]) == 1 and int(k[1]) == 1:
        kernel_size = (1, 1)
    else:
        temp = [1, 1]
        for i, x in enumerate(k):
            if x > 1:
                temp[i] = 3
        kernel_size = tuple(temp)

    return kernel_size


# 若BottleNeckBlock中使用CBAM注意力机制，返回(1, 7)或(7, 1)的kernel_size
def judge_kernel_size_in_cbam(k):
    if int(k[0]) == 1 and int(k[1]) == 1:
        kernel_size = (1, 1)
    else:
        temp = [1, 1]
        for i, x in enumerate(k):
            if x > 1:
                temp[i] = 7
        kernel_size = tuple(temp)

    return kernel_size


# 若BottleNeckBlock中使用SK注意力机制，返回(1, 3), (1, 5)或(3, 1), (5, 1)的kernel_size
def judge_kernel_size_in_sk(k):
    if int(k[0]) == 1 and int(k[1]) == 1:
        kernel_size = [(1, 1), (1, 1)]
    else:
        temp1 = [1, 1]
        temp2 = [1, 1]
        for i, x in enumerate(k):
            if x > 1:
                temp1[i] = 3
                temp2[i] = 5
        kernel_size1 = tuple(temp1)
        kernel_size2 = tuple(temp2)
        kernel_size = [kernel_size1, kernel_size2]
    return kernel_size
