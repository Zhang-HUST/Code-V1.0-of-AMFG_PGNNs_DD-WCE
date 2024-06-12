import torch
import torch.nn as nn

"""
1. SKNet中的SKConv，此版本可代替普通的Conv(in_channels, out_channels): SKConvBlock
2. SKBlock结构：SKConv - Pooling - Dropout:
"""


class SKConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list):
        super().__init__()
        self.CNNList = []
        self.MatrixList = []
        for k in kernel_list:
            # padSize = judge_padSize_in_skconv(k)
            path = nn.Sequential(
                # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1,
                #           padding=padSize),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, stride=1, padding='same'),
                nn.BatchNorm2d(out_channels),
                # nn.LeakyReLU(negative_slope=0.6),
                nn.ReLU(),
            )
            self.CNNList.append(path)
            matrixParam = nn.Parameter(torch.randn(int(out_channels / 2), out_channels))
            self.MatrixList.append(matrixParam)
        self.CNNList = nn.ModuleList(self.CNNList)
        self.MatrixList = nn.ParameterList(self.MatrixList)
        self.linear = nn.Linear(out_channels, int(out_channels / 2))  # 这里的参数设置是随着通道数改变的
        self.softMax4att = nn.Softmax(dim=1)

    def forward(self, x):
        batchSize = x.shape[0]
        outlist = [cnn(x) for cnn in self.CNNList]
        fuse1 = 0
        for out in outlist:
            fuse1 = fuse1 + out
        zVector = nn.AvgPool2d(kernel_size=(x.shape[2], x.shape[3]))(fuse1)  # batchSize * 32 *1 *1
        zVector = zVector.view(batchSize, -1)
        sVector = self.linear(zVector)  # batchSize * 16
        sVector = sVector.unsqueeze(1)
        factorList = [torch.matmul(sVector, matrix) for matrix in self.MatrixList]
        fuseFactor = self.softMax4att(torch.cat(factorList, dim=1))  # batchSize *3* 32
        # 输出各分支注意力系数在各个通道的平均来反映一个分支的注意力水平（尺度注意力水平）
        self.attFactor = torch.mean(fuseFactor, dim=2).detach().to(device='cpu')
        fuseFactor = fuseFactor.unsqueeze(3).unsqueeze(4)
        result = 0
        for k in range(len(self.MatrixList)):
            result = result + fuseFactor[:, k, :] * outlist[k]
        return result


class SKBlock(nn.Module):
    def __init__(self, skconv_dict, pooling_dict, dropout_dict):
        super(SKBlock, self).__init__()
        self.skconv_dict = skconv_dict
        self.pooling_dict = pooling_dict
        self.dropout_dict = dropout_dict

        # sk卷积模块：
        self.skconv_layer = SKConvBlock(in_channels=self.skconv_dict['in_channels'],
                                        out_channels=self.skconv_dict['out_channels'],
                                        kernel_list=self.skconv_dict['kernel_list'])

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
        # SKConv：
        y = self.skconv_layer(x)
        # Pooling:
        y = self.pool(y)
        # Dropout:
        y = self.drop(y)

        return y


def judge_padSize_in_skconv(k):
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
