import torch
import torch.nn as nn

"""
1. Self-Attention: SelfAttBlock
Attention is all you need. NIPS, 2017.
Self-Attention是从NLP中借鉴过来的思想，结构自上而下分为三个分支，分别是query、key和value。
计算时通常分为三步：
第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等；
第二步一般是使用一个softmax函数对这些权重进行归一化；
第三步将权重和相应的键值value进行加权求和得到最后的attention.

2. Non-local Neural Networks: NonLocalAttBlock
Non-local Neural Networks: NonLocalBlock. CVPR 2018.
https://zhuanlan.zhihu.com/p/110130098
卷积网络在统计全局信息时出现的三个问题如下：
1) 捕获长范围特征依赖需要累积很多层的网络，导致学习效率太低；
2) 由于网络需要累计很深，需要小心的设计模块和梯度；
3) 当需要在比较远位置之间来回传递消息时，卷积或者时序局部操作很困难。
基于图片滤波领域的非局部均值滤波操作思想，提出了一个非局部操作算子，可以捕获时间(一维时序信号)、空间(图片)和时空(视频序列)的长范围依赖：
相比较于不断堆叠卷积和RNN算子，非局部操作直接计算两个位置(可以是时间位置、空间位置和时空位置)之间的关系即可快速捕获长范围依赖，
四种实现方式：Gaussian、Embedded Gaussian（下面实现的）、Dot Product、Concatenation

3. CCNet: CCAttBlock
CCNet: Criss-Cross Attention for Semantic Segmentation. ICCV, 2019.

4. GCNet: GCAttBlock
GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond. CVPR, 2019.
"""


# 1. Self-Attention
class SelfAttBlock(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, reduction=8):
        super(SelfAttBlock, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        # return out, attention
        return out


# 2. NonLocal-Attention
class NonLocalAttBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalAttBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=(1, 1),
                                  stride=(1, 1), padding='same', bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=(1, 1),
                                    stride=(1, 1), padding='same', bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=(1, 1),
                                stride=(1, 1), padding='same', bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=(1, 1),
                                   stride=(1, 1), padding='same', bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x

        return out


# 3. CC-Net
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CCAttBlock(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, reduction=8):
        super(CCAttBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


# 4. GC-Net
class GCAttBlock(nn.Module):
    def __init__(self, inplanes, ratio=0.25, pooling_type='att', fusion_types=('channel_add',)):
        # 缩放因子ratio设计：ratio = 1 / 4的时候效果最好
        # 池化和特征融合设计：attention pooling + add的方法效果最好
        super(GCAttBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=(1, 1))
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1)))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1)),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1)))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
