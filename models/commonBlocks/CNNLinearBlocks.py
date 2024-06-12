import torch.nn as nn

"""
    这是一个通用卷积块，其结构由四个Module组成：Conv + BN + Activation + Pooling + Dropout
    每个Module的配置用字典形式实现，字典名不可改
    下面是每个字典应该包括的字段和字段对应的参数形式、含义。
        Conv:
            in_channel      : Conv 的输入通道数
            filters         : Conv 的滤波器个数（定义输出通道数）
            kernel          : (kernel1, kernel2) 定义卷积核大小
            stride          : (stride1, stride2) 定义步长大小
            dilation        : (dilation1, dilation2) 定义膨胀卷积膨胀系数大小 (1,1)表示不用膨胀卷积
            padding         : Conv 的padding方式， 可选：'same' 'valid' , 如果stride不全为1，必须用valid
        *注  意          :  暂不支持自定义padding
        BN :
            use_BN          : True:使用 / False：不使用
        Activation:
            activation_type : 'relu', 'leakyrelu', 'sigmoid', 'None'其他的后面可以加入   #明天加上prelu
            negative_slope  : 控制  LeakyRe LU 负斜率，不是leakyReLU 则该参数不起作用
        Pooling：
            pooling_type    : 'max2d', 'ave2d'
            pooling_kernel  : (kernel1, kernel2) 定义池化核大小
            pooling_stride  : (stride1, stride2) 定义步长大小
        Dropout:
            drop_rate       : 丢失率 0~1

"""


class CNNBlock(nn.Module):
    def __init__(self, conv_dict, bn_dict, activation_dict, pooling_dict, dropout_dict):
        super(CNNBlock, self).__init__()
        self.conv_dict = conv_dict
        self.bn_dict = bn_dict
        self.activation_dict = activation_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict

        # 分别定义四个模块：
        # 卷积模块：
        self.conv_layer = nn.Conv2d(in_channels=self.conv_dict['in_channel'], out_channels=self.conv_dict['filters'],
                                    kernel_size=self.conv_dict['kernel'],
                                    stride=self.conv_dict['stride'], dilation=self.conv_dict['dilation'],
                                    padding=self.conv_dict['padding'])
        # bacth normalization 模块
        self.bn = nn.BatchNorm2d(self.conv_dict['filters'])

        # activation 模块
        if self.activation_dict['activation_type'] == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_dict['activation_type'] == 'leakyrelu':
            self.activation = nn.LeakyReLU(self.activation_dict['negative_slope'])
        elif self.activation_dict['activation_type'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_dict['activation_type'] == 'None':
            self.activation = None
        else:
            raise Exception(f"Not support activation function {self.activation_dict['activation_type']} "
                            f"Only 'relu' 'leakyrelu' and 'sigmoid' can be used, please check")

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
        # Conv：
        y = self.conv_layer(x)
        # BN:
        if self.bn_dict['use_BN']:
            y = self.bn(y)  # BN
        # Activation:
        if self.activation:
            y = self.activation(y)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)
        return y


"""
    这是一个通用线性层： Linear + Activation + Dropout 结构
    参数用字典形式传入，规范如下
    Linear：
        in_dim  :  输入维度
        out_dim :  输出维度
    Activation_linear:
        activation_type : 'relu' , 'leakyrelu', 'sigmoid', 'None'
        negative_slope  : 负斜率
    Dropout_linear:
        drop_rate       : 丢失率 0~1
"""


class LinearBlock(nn.Module):
    def __init__(self, linear_dict, bn_dict, activation_dict, dropout_dict):
        super(LinearBlock, self).__init__()
        self.linear_dict = linear_dict
        self.bn_dict = bn_dict
        self.activation_dict = activation_dict
        self.dropout_dict = dropout_dict
        self.linear = nn.Linear(self.linear_dict['in_dim'], self.linear_dict['out_dim'])
        # bacth normalization 模块
        self.bn = nn.BatchNorm1d(self.linear_dict['out_dim'])

        if self.activation_dict['activation_type'] == 'relu':
            self.activation = nn.ReLU()
        elif self.activation_dict['activation_type'] == 'leakyrelu':
            self.activation = nn.LeakyReLU(self.activation_dict['negative_slope'])
        elif self.activation_dict['activation_type'] == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.activation_dict['activation_type'] == 'None':
            self.activation = None
        else:
            raise Exception(f"Not support activation function {self.activation_dict['activation_type']} "
                            f"Only 'relu' 'leakyrelu' and 'sigmoid' can be used, please check")

        self.drop = nn.Dropout(self.dropout_dict['drop_rate'])

    def forward(self, x):
        y = self.linear(x)
        if self.bn_dict['use_BN']:
            y = self.bn(y)
        if self.activation:
            y = self.activation(y)
        if self.dropout_dict['use_dropout']:
            self.drop(y)
        return y


