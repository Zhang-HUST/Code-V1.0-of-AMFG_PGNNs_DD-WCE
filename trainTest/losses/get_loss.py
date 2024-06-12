import torch
import numpy as np
import torch.nn as nn
from trainTest.losses.focal_loss import FocalLoss


class ClassificationLoss:
    def __init__(self, device, loss_type, callbacks, utils, epoch, class_weights=None):
        self.device = device
        self.loss_type = loss_type
        self.class_weights_list = list(class_weights.detach().double().to(device='cpu').numpy())
        self.class_weights = torch.FloatTensor(class_weights).double().to(self.device)
        # 以下为按照epoch衰减的Weights的方法的初始化
        self.epoch = epoch
        self.max_epoch = callbacks['epoch']
        self.interval = utils['print_interval']
        self.all_modified_weights_list = modify_class_weights_over_epochs(self.class_weights_list, self.max_epoch,
                                                                          self.loss_type['params']['modify_type'],
                                                                          self.loss_type['params']['exponent_factor'])

    def get_criterion(self):
        if self.loss_type['loss_type'] == 'CE':
            if self.epoch == 1:
                print('使用交叉熵损失: ')
            criterion = nn.CrossEntropyLoss()
        elif self.loss_type['loss_type'] == 'WeightedCE':
            if self.epoch == 1:
                print('使用加权交叉熵损失，权重: ', self.class_weights_list)
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif self.loss_type['loss_type'] == 'FL':
            if self.epoch == 1:
                print('使用FocalLoss损失: ')
            criterion = FocalLoss(gamma=2.0)
        elif self.loss_type['loss_type'] == 'WeightedFL':
            if self.epoch == 1:
                print('使用加权FocalLoss损失，权重: ', self.class_weights_list)
            criterion = FocalLoss(gamma=2.0, weight=self.class_weights)
        elif self.loss_type['loss_type'] == 'AttenuationWeightedCE':
            modified_class_weights_list = get_weights_at_epochs(self.all_modified_weights_list, self.epoch)
            if self.epoch == 1 or self.epoch % self.interval == 0:
                print('使用衰减的加权交叉熵损失，权重: ', modified_class_weights_list, '衰减类型：',
                      self.loss_type['params']['modify_type'])
            modified_class_weights = torch.FloatTensor(modified_class_weights_list).double().to(self.device)
            criterion = nn.CrossEntropyLoss(weight=modified_class_weights)
        elif self.loss_type['loss_type'] == 'AttenuationWeightedFL':
            modified_class_weights_list = get_weights_at_epochs(self.all_modified_weights_list, self.epoch)
            if self.epoch == 1 or self.epoch % self.interval == 0:
                print('使用衰减的加权FocalLoss损失，权重: ', modified_class_weights_list, '衰减类型：',
                      self.loss_type['params']['modify_type'])
            modified_class_weights = torch.FloatTensor(modified_class_weights_list).double().to(self.device)
            criterion = FocalLoss(gamma=2.0, weight=modified_class_weights)
        else:
            raise ValueError('loss_type must be one of [CE, WeightedCE, AttenuationWeightedCE, '
                             'FL, WeightedFL, AttenuationWeightedFL]')

        return criterion


def modify_class_weights_over_epochs(class_weights, epochs, modify_type, exponent_factor):
    modified_weights_list = []

    for weight in class_weights:
        if weight == 1:
            modified_weights = [1] * epochs
        elif weight < 1:
            if modify_type == 'linear':
                modified_weights = [min(1, weight + (1 - weight) * i / epochs) for i in range(1, epochs + 1)]
            elif modify_type == 'exponent':
                modified_weights = [min(1, weight + (1 - weight) * (1 - np.exp(-exponent_factor * i / epochs))) for i in
                                    range(1, epochs + 1)]
            else:
                raise ValueError('modify_type must be one of [linear, exponent]')
        else:
            if modify_type == 'linear':
                modified_weights = [max(1, weight - (weight - 1) * i / epochs) for i in range(1, epochs + 1)]
            elif modify_type == 'exponent':
                modified_weights = [max(1, weight - (weight - 1) * (1 - np.exp(-exponent_factor * i / epochs))) for i in
                                    range(1, epochs + 1)]
            else:
                raise ValueError('modify_type must be one of [linear, exponent]')

        modified_weights_list.append(modified_weights)

    return modified_weights_list


def get_weights_at_epochs(modified_weights_list, epoch):
    weights_at_epoch = []
    if 1 <= epoch <= len(modified_weights_list[0]):
        weights_at_epoch = [weights[epoch - 1] for weights in modified_weights_list]

    return weights_at_epoch


"""
import matplotlib.pyplot as plt
class_weights = [0.5, 1, 2]
max_epoch = 100
new_weights_list = modify_class_weights_over_epochs(class_weights, max_epoch)
fig, axes = plt.subplots(nrows=len(class_weights), ncols=1, figsize=(8, 10))
for i in range(len(class_weights)):
     axes[i].plot(new_weights_list[i])
"""