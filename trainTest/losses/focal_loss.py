# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""FocalLoss，以及仿照平衡交叉熵损失实现的平衡FocalLoss"""

# 普通FocalLoss，不指定alpha
# criterion = FocalLoss(gamma=2.0, weight=None)
# alpha是平衡Focal Loss中的类别权重参数，用于调整每个类别的贡献度。它可以根据数据集的类别不平衡情况来进行设置。
# 以下是一些设置alpha的常见方法：
# 1. 手动设置权重：根据数据集的先验知识或经验，手动设置每个类别的权重 alpha = torch.tensor([0.5, 1.0, 0.8])
# 2. 根据误分类成本设置权重：根据对模型错误分类的成本敏感程度，设置每个类别的权重。对于那些错误分类成本较高的类别，可以分配较大的权重。
# cost_matrix = ... alpha = cost_matrix[target]
# 3. 相反类频率权重（Inverse Class Frequency Weighting）：计算每个类别在训练集中的样本数量，并将总样本数除以每个类别的样本数来得到类别权重，如果某个类别有较少的样本，则其权重较大。
# num_classes = 5  # 样本的类别数，可不设置
# from utils.universal_tools.utils import calculate_class_weights_torch, calculate_class_weights_sklearn
# class_weights = calculate_class_weights_torch(label, num_classes=num_classes) # num_classes可不设置
# criterion = FocalLoss(gamma=2.0, weight=class_weights)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = None

    def forward(self, input, target):
        # 计算交叉熵损失
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')

        # 计算Focal Loss
        focal_weights = torch.pow(1 - torch.softmax(input, dim=1), self.gamma)

        if self.weight is None:
            focal_loss = focal_weights * ce_loss
            loss = focal_loss.mean()
        else:
            self.alpha = self.weight[input]
            balanced_focal_loss = self.alpha * focal_weights * ce_loss
            loss = balanced_focal_loss.mean()
        return loss



"""下面是FocalLoss的其他实现方式"""


class FocalLoss1(nn.Module):
    def __init__(self, gamma=2, alpha=1, size_average=True):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert (logits.size(0) == labels.size(0))
        assert (logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()


class FocalLoss2(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class FocalLoss3(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss3, self).__init__()

        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):

        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss

"""据说是加权的FocalLoss"""
class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(inputs, targets)

        # 计算分类概率
        prob = torch.softmax(inputs, dim=1)

        # 获取每个样本的类别预测概率
        p_t = prob[torch.arange(len(targets)), targets]

        # 根据类别数量调整权重因子
        if self.alpha is not None:
            alpha = self.alpha[targets]
            weighted_ce_loss = torch.mul(alpha, ce_loss)
        else:
            weighted_ce_loss = ce_loss

        # 计算加权焦点损失
        focal_loss = torch.mul(torch.pow(1 - p_t, self.gamma), weighted_ce_loss)

        # 求平均损失
        loss = torch.mean(focal_loss)

        return loss


# # 假设有 10 个类别
# num_classes = 10
#
# # 创建加权焦点损失函数实例
# weighted_focal_loss = WeightedFocalLoss(alpha=torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), gamma=2)
#
# # 示例输入和目标
# inputs = torch.randn(64, num_classes)  # 模型的预测输出
# targets = torch.randint(0, num_classes, (64,))  # 真实标签
#
# # 计算损失
# loss = weighted_focal_loss(inputs, targets)


