import torch
import torch.nn as nn
from models.commonBlocks.CNNLinearBlocks import CNNBlock, LinearBlock
from models.commonBlocks.SKConvBlock import SKBlock
from models.commonBlocks.ResNetBlocks import ResNetBlock
from models.commonBlocks.MobileNetBlocks import MobileNetBlock
from models.commonBlocks.ResNeStBlock import ResNeStBlock
from models.commonBlocks.ShffleNetBlocks import ShuffleNetBlock
from models.configs.CNNsConfigs import (common_configs, cnn_configs, sk_configs, resnet_configs, mobilenet_configs,
                                        resnest_configs, shfflenet_configs, linear_configs)



# 输入：data: batch,1,15,96, feature: batch,15, 6, conv_type: one of [Conv, SKConv]
# 输出：batch*num_classes
class GeneralCNNs(nn.Module):
    def __init__(self, network_branch, gait_or_motion, motion_type, conv_type='Conv'):
        super(GeneralCNNs, self).__init__()
        self.network_branch, self.conv_type = network_branch, conv_type
        bascic_configs = common_configs()
        (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
         Dropout3, BN4, Activation4, Pooling4, Dropout4, BN5, Activation5, Pooling5, Dropout5) = bascic_configs

        fc_configs = linear_configs(gait_or_motion, motion_type, conv_type, network_branch)
        (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
         Dropout2_linear) = fc_configs

        if self.conv_type == 'Conv':
            configs = cnn_configs()
            (Conv1, Conv2, Conv3, Conv4, Conv5) = configs
            self.cnn_part1 = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                CNNBlock(conv_dict=Conv2, bn_dict=BN2, activation_dict=Activation2, pooling_dict=Pooling2,
                         dropout_dict=Dropout2),
                CNNBlock(conv_dict=Conv3, bn_dict=BN3, activation_dict=Activation3, pooling_dict=Pooling3,
                         dropout_dict=Dropout3),
            )
            self.cnn_part2 = nn.Sequential(
                CNNBlock(conv_dict=Conv4, bn_dict=BN4, activation_dict=Activation4, pooling_dict=Pooling4,
                         dropout_dict=Dropout4),
                CNNBlock(conv_dict=Conv5, bn_dict=BN5, activation_dict=Activation5, pooling_dict=Pooling5,
                         dropout_dict=Dropout5), )

        elif self.conv_type == 'SKConv':
            configs = sk_configs()
            (SKConv1, SKConv2, SKConv3, SKConv4, SKConv5) = configs
            self.cnn_part1 = nn.Sequential(
                SKBlock(skconv_dict=SKConv1, pooling_dict=Pooling1, dropout_dict=Dropout1),
                SKBlock(skconv_dict=SKConv2, pooling_dict=Pooling2, dropout_dict=Dropout2),
                SKBlock(skconv_dict=SKConv3, pooling_dict=Pooling3, dropout_dict=Dropout3),
            )
            self.cnn_part2 = nn.Sequential(
                SKBlock(skconv_dict=SKConv4, pooling_dict=Pooling4, dropout_dict=Dropout4),
                SKBlock(skconv_dict=SKConv5, pooling_dict=Pooling5, dropout_dict=Dropout5), )

        elif self.conv_type in ['ResNetV1', 'ResNetV2']:
            configs = resnet_configs()
            (Conv1, ResNet_2, ResNet_3, ResNet_4, ResNet_5) = configs
            self.cnn_part1 = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ResNetBlock(resnet_dict=ResNet_2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                            block_type=self.conv_type),
                ResNetBlock(resnet_dict=ResNet_3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                            block_type=self.conv_type),
            )
            self.cnn_part2 = nn.Sequential(
                ResNetBlock(resnet_dict=ResNet_4, pooling_dict=Pooling4, dropout_dict=Dropout4,
                            block_type=self.conv_type),
                ResNetBlock(resnet_dict=ResNet_5, pooling_dict=Pooling5, dropout_dict=Dropout5,
                            block_type=self.conv_type),
            )

        elif self.conv_type in ['MobileNetV1', 'MobileNetV2', 'MobileNetV3']:
            configs = mobilenet_configs()
            (Conv1, MobileNet_2, MobileNet_3, MobileNet_4, MobileNet_5) = configs
            self.cnn_part1 = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                MobileNetBlock(mobilenet_dict=MobileNet_2, bn_dict=BN2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                               block_type=self.conv_type),
                MobileNetBlock(mobilenet_dict=MobileNet_3, bn_dict=BN3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                               block_type=self.conv_type),
            )
            self.cnn_part2 = nn.Sequential(
                MobileNetBlock(mobilenet_dict=MobileNet_4, bn_dict=BN4, pooling_dict=Pooling4, dropout_dict=Dropout4,
                               block_type=self.conv_type),
                MobileNetBlock(mobilenet_dict=MobileNet_5, bn_dict=BN5, pooling_dict=Pooling5, dropout_dict=Dropout5,
                               block_type=self.conv_type))

        elif self.conv_type == 'ResNeSt':
            configs = resnest_configs()
            (Conv1, ResNeSt_2, ResNeSt_3, ResNeSt_4, ResNeSt_5) = configs
            self.cnn_part1 = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ResNeStBlock(resnestconv_dict=ResNeSt_2, pooling_dict=Pooling2, dropout_dict=Dropout2),
                ResNeStBlock(resnestconv_dict=ResNeSt_3, pooling_dict=Pooling3, dropout_dict=Dropout3),
            )
            self.cnn_part2 = nn.Sequential(
                ResNeStBlock(resnestconv_dict=ResNeSt_4, pooling_dict=Pooling4, dropout_dict=Dropout4),
                ResNeStBlock(resnestconv_dict=ResNeSt_5, pooling_dict=Pooling5, dropout_dict=Dropout5), )

        elif self.conv_type in ['ShuffleNetV1', 'ShuffleNetV2']:
            configs = shfflenet_configs()
            (Conv1, ShuffleNet_2, ShuffleNet_3, ShuffleNet_4, ShuffleNet_5) = configs
            self.cnn_part1 = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                                block_type=self.conv_type),
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                                block_type=self.conv_type),
            )
            self.cnn_part2 = nn.Sequential(
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_4, pooling_dict=Pooling4, dropout_dict=Dropout4,
                                block_type=self.conv_type, branch=network_branch),
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_5, pooling_dict=Pooling5, dropout_dict=Dropout5,
                                block_type=self.conv_type), )
        elif self.conv_type == 'DNN':
            self.linear_part = nn.Sequential(
                LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                            dropout_dict=Dropout1_linear),
                LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                            dropout_dict=Dropout2_linear),
            )
        else:
            raise ValueError('conv_type must be one of [DNN, Conv, SKConv, ResNetV1, ResNetV2, MobileNetV1, MobileNetV2, '
                             'MobileNetV3, ResNeSt, ShuffleNetV1, ShuffleNetV2]')

        self.linear_part = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                        dropout_dict=Dropout2_linear),
        )
        self.init_params()

    def forward(self, data, feature):
        # data: [32, 1, 15, 96], feature: [32, 1, 15, 6]
        if self.conv_type == 'DNN':
            new_feature = feature.view(*(feature.size(0), -1))
            out = self.linear_part(new_feature)
        else:
            batch_size = data.shape[0]
            # [32, 96, 15, 1]
            cnn_out_1 = self.cnn_part1(data)
            # [32, 256, 1, 1]
            cnn_out_2 = self.cnn_part2(cnn_out_1)
            # [32, 256]
            cnn_out_2 = cnn_out_2.reshape(batch_size, -1)
            if self.network_branch == 1:
                # torch.Size([32, 256])
                y = cnn_out_2
            elif self.network_branch == 2:
                # # 在最后增加一个维度, 将第二维和第四维交换位置
                # # feature = feature.unsqueeze(3)
                # # [32, 6, 15, 1]
                # feature = torch.transpose(feature, 1, 3).contiguous()
                # # [32, 96+6, 15, 1]
                # # print(feature.size(), cnn_out_1.size())
                # y = torch.cat([cnn_out_1, feature], dim=1)
                new_feature = feature.view(*(feature.size(0), -1))
                # torch.Size([32, 256+90])
                y = torch.cat([cnn_out_2, new_feature], dim=1)
            else:
                raise ValueError('network_branch wrong! support 1 and 2 only!')
            out = self.linear_part(y)

        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                # mode选择“fan_in”保留了向前传递中权重方差的大小。选择“fan_out”保留反向传递的大小
                # 'fan_in'（默认值）：适用于使用ReLU激活函数的网络层。这个模式考虑了权重矩阵的输入单元数量（fan - in），并根据它来初始化权重。对ReLU激活函数友好。
                # 'fan_out'：适用于使用带有LeakyReLU或sigmoid等不是很常见的激活函数的网络层。这个模式考虑了权重矩阵的输出单元数量（fan - out），并根据它来初始化权重。
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_name(self):
        if self.conv_type == 'DNN':
            return 'OneBranchDNN'
        else:
            if self.network_branch == 1:
                basic_name = 'OneBranchCNN-'
            elif self.network_branch == 2:
                basic_name = 'TwoBranchCNN-'
            else:
                raise ValueError('network_branch wrong ! support 1 and 2 only!')
            return basic_name + self.conv_type
