from utils.common_utils import get_num_classes


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

    BN4 = {'use_BN': True}
    # Activation4 = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    Activation4 = {'activation_type': 'relu'}
    Pooling4 = {'pooling_type': 'max2d', 'pooling_kernel': (5, 1), 'pooling_stride': (5, 1)}
    Dropout4 = {'drop_rate': 0.2}

    BN5 = {'use_BN': True}
    # Activation5 = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    Activation5 = {'activation_type': 'relu'}
    Pooling5 = {'pooling_type': 'ave2d', 'pooling_kernel': (3, 1), 'pooling_stride': (1, 1)}
    Dropout5 = {'drop_rate': 0.2}

    configs = (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
               Dropout3, BN4, Activation4, Pooling4, Dropout4, BN5, Activation5, Pooling5, Dropout5)

    return configs


def cnn_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv2 = {'in_channel': 32, 'filters': 64, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv3 = {'in_channel': 64, 'filters': 96, 'kernel': (1, 3), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv4 = {'in_channel': 96, 'filters': 128, 'kernel': (3, 1), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    Conv5 = {'in_channel': 128, 'filters': 256, 'kernel': (3, 1), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}

    configs = (Conv1, Conv2, Conv3, Conv4, Conv5)

    return configs


def sk_configs():
    SKConv1 = {'in_channels': 1, 'out_channels': 32, 'kernel_list': [(1, 3), (1, 5), (1, 7)]}
    SKConv2 = {'in_channels': 32, 'out_channels': 64, 'kernel_list': [(1, 3), (1, 5), (1, 7)]}
    SKConv3 = {'in_channels': 64, 'out_channels': 96, 'kernel_list': [(1, 3), (1, 5)]}
    SKConv4 = {'in_channels': 96, 'out_channels': 128, 'kernel_list': [(3, 1), (5, 1)]}
    SKConv5 = {'in_channels': 128, 'out_channels': 256, 'kernel_list': [(3, 1), (1, 1)]}

    configs = (SKConv1, SKConv2, SKConv3, SKConv4, SKConv5)

    return configs


def resnet_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ResNet_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'stride': (1, 1), 'padding': 'same'}
    ResNet_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'stride': (1, 1), 'padding': 'same'}
    ResNet_4 = {'in_channels': 96, 'out_channels': 128, 'kernel': (3, 1), 'stride': (1, 1),
                'padding': 'same'}
    ResNet_5 = {'in_channels': 128, 'out_channels': 256, 'kernel': (3, 1), 'stride': (1, 1),
                'padding': 'same'}

    configs = (Conv1, ResNet_2, ResNet_3, ResNet_4, ResNet_5)

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
    MobileNet_4 = {'in_channels': 96, 'out_channels': 128, 'kernel_size': (3, 1), 'stride': (1, 1),
                   'factor': 2, 'block_num': 1, 'expand_size': 96 * 2, 'nonlinear': 'HardSwish',
                   'attention': 'SE'}
    MobileNet_5 = {'in_channels': 128, 'out_channels': 256, 'kernel_size': (3, 1), 'stride': (1, 1),
                   'factor': 2, 'block_num': 1, 'expand_size': 128 * 2, 'nonlinear': 'HardSwish', 'attention': 'None'}

    configs = (Conv1, MobileNet_2, MobileNet_3, MobileNet_4, MobileNet_5)

    return configs


def resnest_configs():
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ResNeSt_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'radix': 2, 'groups': 1}
    ResNeSt_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'radix': 2, 'groups': 1}
    ResNeSt_4 = {'in_channels': 96, 'out_channels': 128, 'kernel': (3, 1), 'radix': 2, 'groups': 1}
    ResNeSt_5 = {'in_channels': 128, 'out_channels': 256, 'kernel': (3, 1), 'radix': 2, 'groups': 1}

    configs = (Conv1, ResNeSt_2, ResNeSt_3, ResNeSt_4, ResNeSt_5)

    return configs


def shfflenet_configs():
    # group = 4 if network_branch == 1 else 1
    group = 1
    # batch*32*15*96
    Conv1 = {'in_channel': 1, 'filters': 32, 'kernel': (1, 5), 'stride': (1, 1), 'dilation': (1, 1),
             'padding': 'same'}
    ShuffleNet_2 = {'in_channels': 32, 'out_channels': 64, 'kernel': (1, 3), 'groups': 1}
    ShuffleNet_3 = {'in_channels': 64, 'out_channels': 96, 'kernel': (1, 3), 'groups': 1}
    ShuffleNet_4 = {'in_channels': 96, 'out_channels': 128, 'kernel': (3, 1), 'groups': group}
    ShuffleNet_5 = {'in_channels': 128, 'out_channels': 256, 'kernel': (3, 1), 'groups': 1}

    configs = (Conv1, ShuffleNet_2, ShuffleNet_3, ShuffleNet_4, ShuffleNet_5)

    return configs


def linear_configs(gait_or_motion, motion_type, conv_type, network_branch):
    num_classes = get_num_classes(gait_or_motion, motion_type)
    if conv_type == 'DNN':
        linear1_in_dim = int(15 * 6)
    else:
        linear1_in_dim = 256 if network_branch == 1 else int(256 + 15 * 6)

    Linear1 = {'in_dim': linear1_in_dim, 'out_dim': 64}
    # Activation1_linear = {'activation_type': 'leakyrelu', 'negative_slope': 0.3}
    BN1_linear = {'use_BN': True}
    Activation1_linear = {'activation_type': 'relu'}
    Dropout1_linear = {'use_dropout': True, 'drop_rate': 0.2}

    Linear2 = {'in_dim': 64, 'out_dim': num_classes}
    BN2_linear = {'use_BN': False}
    Activation2_linear = {'activation_type': 'None'}
    Dropout2_linear = {'use_dropout': False, 'drop_rate': 0.2}

    configs = (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
               Dropout2_linear)

    return configs
