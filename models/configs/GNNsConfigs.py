import os
import torch
import numpy as np
import pandas as pd
from utils.common_utils import get_num_classes


# GNNs相关配置
def gnn_configs(psk_path):
    # GNN部分
    # cheb_K: 一般为2~3；'ssg_alpha': Teleport probability [0, 1],论文中最好0.05, 'ssg_K': 节点矩阵聚集的阶数，即层数，论文中最好16}
    GCNs1 = {'in_channels': 96, 'out_channels': 128, 'cheb_K': 3, 'ssg_alpha': 0.05, 'ssg_K': 16}
    GCNs2 = {'in_channels': 128, 'out_channels': 128, 'cheb_K': 3, 'ssg_alpha': 0.05, 'ssg_K': 16}
    GATs1 = {'in_channels': 96, 'out_channels': 64, 'heads': 2, 'concat': True}
    GATs2 = {'in_channels': 128, 'out_channels': 128, 'heads': 2, 'concat': False}

    psk_matrix, psk_edge_num = get_psk_matrix(psk_path)
    configs = (GCNs1, GCNs2, GATs1, GATs2, psk_matrix, psk_edge_num)

    return configs


def linear_configs(gait_or_motion, motion_type, gnn_out_dim, network_branch, node_amount, readout_mode):
    num_classes = get_num_classes(gait_or_motion, motion_type)
    if readout_mode == 'fc':
        if network_branch == 1:
            linear1_in_dim = int(gnn_out_dim * node_amount)
        elif network_branch == 2:
            linear1_in_dim = int(gnn_out_dim * node_amount + 15 * 6)
        else:
            raise ValueError('network_branch wrong! support 1 and 2 only!')
    elif readout_mode in ['mean', 'max']:
        if network_branch == 1:
            linear1_in_dim = gnn_out_dim
        elif network_branch == 2:
            linear1_in_dim = int(gnn_out_dim + 15 * 6)
        else:
            raise ValueError('network_branch wrong! support 1 and 2 only!')
    else:
        raise ValueError('readout_mode wrong! support mean, max and fc only!')

    Linear1 = {'in_dim': linear1_in_dim, 'out_dim': 32}
    # Activation1_linear = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    BN1_linear = {'use_BN': True}
    Activation1_linear = {'activation_type': 'relu'}
    Dropout1_linear = {'use_dropout': True, 'drop_rate': 0.2}

    Linear2 = {'in_dim': 32, 'out_dim': num_classes}
    BN2_linear = {'use_BN': False}
    Activation2_linear = {'activation_type': 'None'}
    Dropout2_linear = {'use_dropout': False, 'drop_rate': 0.2}

    configs = (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
               Dropout2_linear)

    return configs


def get_psk_matrix(psk_path):
    file_name = os.path.join(psk_path, 'PSK_matrix.csv')
    df = pd.read_csv(file_name, header=0, index_col=0)
    psk = torch.from_numpy(df.values)
    psk_edge_num = np.sum(df.values == 1)
    # psk_edge_num = np.count_nonzero(psk == 1)

    return psk, psk_edge_num


# CNNs相关配置
def common_configs():
    BN1 = {'use_BN': True}
    # Activation1 = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    Activation1 = {'activation_type': 'relu'}
    Pooling1 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout1 = {'drop_rate': 0.2}

    BN2 = {'use_BN': True}
    # Activation2 = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    Activation2 = {'activation_type': 'relu'}
    Pooling2 = {'pooling_type': 'max2d', 'pooling_kernel': (1, 4), 'pooling_stride': (1, 4)}
    Dropout2 = {'drop_rate': 0.2}

    BN3 = {'use_BN': True}
    Activation3 = {'activation_type': 'relu'}
    Pooling3 = {'pooling_type': 'ave2d', 'pooling_kernel': (1, 6), 'pooling_stride': (1, 1)}
    Dropout3 = {'drop_rate': 0.2}

    configs = (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
               Dropout3)

    return configs


def cnn_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv2 = {'in_channel': 32, 'filters': 64, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv3 = {'in_channel': 64, 'filters': 96, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}

    configs = (Conv1, Conv2, Conv3)

    return configs


def sk_configs():
    SKConv1 = {'in_channels': 1, 'out_channels': 32, 'kernel_list': [(1, 3), (1, 5), (1, 7)]}
    SKConv2 = {'in_channels': 32, 'out_channels': 64, 'kernel_list': [(1, 3), (1, 5), (1, 7)]}
    SKConv3 = {'in_channels': 64, 'out_channels': 96, 'kernel_list': [(1, 3), (1, 5)]}

    configs = (SKConv1, SKConv2, SKConv3)

    return configs


def resnet_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ResNet_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'stride': (1, 1), 'padding': 'same'}
    ResNet_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'stride': (1, 1), 'padding': 'same'}

    configs = (Conv1, ResNet_2, ResNet_3)

    return configs


def mobilenet_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    # 'channel attention': ['SE', 'SK', 'ECA', 'Split']
    # 'spatial attention': ['SelfAtt', 'NonLocal', 'CC', 'GC']
    # 'channel spatial attention': ['BAM', 'CBAM', 'DA',  'CA', 'SA', 'Triplet', 'SC']
    MobileNet_2 = {'in_channels': 32, 'out_channels': 64, 'kernel_size': (1, 3), 'stride': (1, 1),
                   'factor': 1, 'block_num': 1, 'expand_size': 64, 'nonlinear': 'ReLU', 'attention': 'None'}
    MobileNet_3 = {'in_channels': 64, 'out_channels': 96, 'kernel_size': (1, 3), 'stride': (1, 1),
                   'factor': 2, 'block_num': 1, 'expand_size': 64 * 2, 'nonlinear': 'ReLU', 'attention': 'SE'}

    configs = (Conv1, MobileNet_2, MobileNet_3)

    return configs


def resnest_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ResNeSt_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'radix': 2, 'groups': 1}
    ResNeSt_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'radix': 2, 'groups': 1}

    configs = (Conv1, ResNeSt_2, ResNeSt_3)

    return configs


def shfflenet_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ShuffleNet_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'groups': 1}
    ShuffleNet_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'groups': 1}

    configs = (Conv1, ShuffleNet_2, ShuffleNet_3)

    return configs
