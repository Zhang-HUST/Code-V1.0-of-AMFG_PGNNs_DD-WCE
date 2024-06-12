import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.commonBlocks.ActivationFunctions import HardSigmoid

"""
1. 1) 原始的SE-Net: SEAttBlockOriginal, 使用sigmoid激活
      Squeeze-and-excitation networks. CVPR, 2018.
   2) MobileNetV3中的SE-Net: SEAttBlock, 使用HardSigmoid代替sigmoid
      Searching for mobilenetv3. ICCV, 2019.
      
2. SK-Net: SKAttBlock
Selective Kernel Networks. CVPR, 2019.

3. ECA-Net: ECAAttBlock
ECA-Net: Efficient channel attention for deep convolutional neural networks. CVPR, 2020.

4. ResNeSt: Split-Attention Networks中的注意力机制，SplitAttBlock
"""


# 1.1 原始的SE-Net: 使用sigmoid激活
class SEAttBlockOriginal(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEAttBlockOriginal, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


# 1.2 MobileNetV3中的SE-Net: 使用HardSigmoid代替sigmoid
class SEAttBlock(nn.Module):
    def __init__(self, in_channel, activation='hard-sigmoid', reduction=4):
        super(SEAttBlock, self).__init__()
        if activation == 'sigmoid':
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(in_channel // reduction),
                nn.ReLU(),
                nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(in_channel),
                nn.Sigmoid(),
            )
        elif activation == 'hard-sigmoid':
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(in_channel // reduction),
                nn.ReLU(),
                nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(in_channel),
                HardSigmoid(),
            )
        else:
            raise ValueError('Unknown activation function {}'.format(activation))

    def forward(self, x):
        return x * self.se(x)


# 2 作为注意力机制的SK-Net: SKBlock
class SKAttBlock(nn.Module):
    def __init__(self, features, kernel_list, G=32, r=16, stride=(1, 1), L=32):
        super(SKAttBlock, self).__init__()
        M = len(kernel_list)
        d = max(int(features / r), L)
        self.features = features
        self.convs = nn.ModuleList([])
        for kernel in kernel_list:
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=kernel, stride=stride, padding='same', groups=G),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feas, attention_vectors = None, None
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


# 3. ECA-Net
class ECAAttBlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
    """

    def __init__(self, channel, gamma=2, b=1):
        super(ECAAttBlock, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        k_size = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# 4. ResNeSt-Net: Split Attention
class rSoftMax(nn.Module):
    """
    (radix-majorize) softmax class

    input is cardinal-major shaped tensor.
    transpose to radix-major
    """

    def __init__(self, groups=1, radix=2):
        super(rSoftMax, self).__init__()

        self.groups = groups
        self.radix = radix

    def forward(self, x):
        B = x.size(0)
        # transpose to radix-major
        x = x.view(B, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(B, -1, 1, 1)

        return x


class SplitAttBlock(nn.Module):
    """
    split attention class
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding='same', dilation=(1, 1), groups=1,
                 bias=True, radix=2, reduction_factor=4):
        super(SplitAttBlock, self).__init__()

        self.radix = radix

        self.radix_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels * radix, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups * radix, bias=bias),
            nn.BatchNorm2d(channels * radix),
            nn.ReLU()
        )

        inter_channels = max(32, in_channels * radix // reduction_factor)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=(1, 1), groups=groups),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=inter_channels, out_channels=channels * radix, kernel_size=1, groups=groups)
        )

        self.rsoftmax = rSoftMax(groups=groups, radix=radix)

    def forward(self, x):
        # NOTE: comments are ugly...

        """
        input  : |             in_channels               |
        """

        '''
        radix_conv : |                radix 0            |               radix 1             | ... |                radix r            |
                     | group 0 | group 1 | ... | group k | group 0 | group 1 | ... | group k | ... | group 0 | group 1 | ... | group k |
        '''
        x = self.radix_conv(x)

        '''
        split :  [ | group 0 | group 1 | ... | group k |,  | group 0 | group 1 | ... | group k |, ... ]

        sum   :  | group 0 | group 1 | ...| group k |
        '''
        B, rC = x.size()[:2]
        splits = torch.split(x, rC // self.radix, dim=1)
        gap = sum(splits)

        '''
        !! becomes cardinal-major !!
        attention : |             group 0              |             group 1              | ... |              group k             |
                    | radix 0 | radix 1| ... | radix r | radix 0 | radix 1| ... | radix r | ... | radix 0 | radix 1| ... | radix r |
        '''
        att_map = self.attention(gap)

        '''
        !! transposed to radix-major in rSoftMax !!
        rsoftmax : same as radix_conv
        '''
        att_map = self.rsoftmax(att_map)

        '''
        split : same as split
        sum : same as sum
        '''
        att_maps = torch.split(att_map, rC // self.radix, dim=1)
        out = sum([att_map * split for att_map, split in zip(att_maps, splits)])

        '''
        output : | group 0 | group 1 | ...| group k |

        concatenated tensors of all groups,
        which split attention is applied
        '''

        return out.contiguous()
