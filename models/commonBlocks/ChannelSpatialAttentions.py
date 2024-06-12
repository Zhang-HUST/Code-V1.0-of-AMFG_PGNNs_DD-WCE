import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.commonBlocks.ActivationFunctions import HardSigmoid, HardSwish

"""
1. BAM-Net: BAMAttBlock
Bam: Bottleneck attention module[J]. arXiv preprint arXiv:1807.06514, 2018.

2. CBAM-Net: CBAMAttBlock
Cbam: Convolutional block attention module. ECCV, 2018.

3. DA-Net:  简单的DA-Net -- DAAttBlock, 完整的DA-Net -- DAHeadAttBlock
Dual attention network for scene segmentation. CVPR, 2019.

4. CA-Net: CAAttBlock
Coordinate attention for efficient mobile network design. CVPR, 2021.

5. SA-Net: SAAttBlock
Sa-net: Shuffle attention for deep convolutional neural networks. ICASSP, 2021.

6. Triplet-Net: TripletAttBlock
Rotate to attend: Convolutional triplet attention module. WACV, 2021.

7. SC-Net: SCAttBlock
SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy and Pattern Recognition. CVPR, 2023.
"""


# BAM-Net和CBAM-Net
def conv1(in_channels, out_channels, stride=(1, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same', stride=stride, bias=False)


def conv3(in_channels, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same',
                     dilation=dilation, bias=False)


def conv7(in_channels, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='same',
                     dilation=dilation, bias=False)


# 1. BAM-Net
class BAMAttBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, reduction_ratio=16, dilation=(1, 1)):
        super(BAMAttBlock, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        self.bn2_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = conv1(in_channel, self.hid_channel)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv2 = conv3(self.hid_channel, self.hid_channel, kernel_size=kernel_size, stride=(1, 1),
                           dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv3 = conv3(self.hid_channel, self.hid_channel, kernel_size=kernel_size, stride=(1, 1),
                           dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv4 = conv1(self.hid_channel, 1)
        self.bn4_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf


# 2. CBAM-Net
class CBAMAttBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, reduction_ratio=16, dilation=(1, 1)):
        super(CBAMAttBlock, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv7(2, 1, kernel_size=kernel_size, stride=(1, 1),
                           dilation=self.dilation)

    def forward(self, x):
        # Channel attention.
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)
        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc * x

        # Spatial attention.
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1
        return Mf2


# 3. DA-Net
class PositionAttentionModuleDA(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, reduction=8, **kwargs):
        super(PositionAttentionModuleDA, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(1, 1), padding='same')
        self.conv_c = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=(1, 1), padding='same')
        self.conv_d = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding='same')
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class ChannelAttentionModuleDA(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(ChannelAttentionModuleDA, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out


# 3.1 简单的DA-Net
class DAAttBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(DAAttBlock, self).__init__()
        self.pam = PositionAttentionModuleDA(in_channels=in_channels, reduction=reduction)
        self.cam = ChannelAttentionModuleDA()

    def forward(self, x):
        feat_p = self.pam(x)
        feat_c = self.cam(x)
        return feat_p + feat_c


# 3.2 完整的的DA-Net
class DAHeadAttBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction=4, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DAHeadAttBlock, self).__init__()
        inter_channels = in_channels // reduction
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size, padding='same', bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size, padding='same', bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        self.pam = PositionAttentionModuleDA(inter_channels, reduction=8, **kwargs)
        self.cam = ChannelAttentionModuleDA(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size, padding='same', bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size, padding='same', bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU()
        )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        return feat_fusion


# 4. CA-Net
class CAAttBlock(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CAAttBlock, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HardSwish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=(1, 1), stride=(1, 1), padding='same')
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = identity * a_w * a_h
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out


# 5. SA-Net
class SAAttBlock(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(SAAttBlock, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


# 6. Triplet-Net
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialGate, self).__init__()

        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=(1, 1), padding='same'),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttBlock(nn.Module):
    def __init__(self, kernel_size, spatial=True):
        super(TripletAttBlock, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate(kernel_size)
        self.width_gate = SpatialGate(kernel_size)
        if self.spatial:
            self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)


# 7. SC-Net
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    # alpha: 0<alpha<1

    def __init__(self,
                 op_channel: int,
                 group_kernel_size,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=(1, 1), padding='same',
                                  bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=(1, 1), padding='same',
                                  bias=False)
        # up
        # self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=(1, 1),
        #                      padding=group_kernel_size // 2, groups=group_size)
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=(1, 1),
                             padding='same', groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=(1, 1), padding='same', bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio,
                              kernel_size=(1, 1), padding='same', bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class SCAttBlock(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_kernel_size,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 ):
        super(SCAttBlock, self).__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

