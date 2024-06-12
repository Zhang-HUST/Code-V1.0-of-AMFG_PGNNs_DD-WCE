import torch
import numpy as np
import torch.nn as nn
from hmmlearn import hmm


class HMMGaitClassification:
    def __init__(self, model, train_loader, test_loader, true_labels, gait_dataset_divide_mode, num_classes=None):
        self.model = model.to(device='cpu')
        self.train_outputs, self.true_train_labels = self.get_feature_labels_from_dataloader(train_loader)
        self.test_outputs, self.true_test_labels = self.get_feature_labels_from_dataloader(test_loader)
        self.true_labels = true_labels
        self.gait_dataset_divide_mode = gait_dataset_divide_mode
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = np.unique(self.true_train_labels)
        self.init()

    def get_feature_labels_from_dataloader(self, data_loader):
        deep_features_list = []
        label_list = []
        for data, feature, label in data_loader:
            deep_feature = self.model(data, feature)
            deep_feature = nn.Softmax(dim=1)(deep_feature)
            deep_features_list.append(deep_feature)
            label_list.append(label)
        deep_features = torch.cat(deep_features_list, dim=0).detach().numpy()
        labels = torch.cat(label_list, dim=0).view(-1).numpy()

        return deep_features, labels

    def init(self):
        transmat_labels = None
        if self.gait_dataset_divide_mode == 'random':
            transmat_labels = self.true_test_labels
        elif self.gait_dataset_divide_mode == 'group_fix':
            transmat_labels = self.true_test_labels
            # transmat_labels = self.true_labels
        elif self.gait_dataset_divide_mode == 'group_random':
            # transmat_labels = self.true_test_labels
            transmat_labels = self.true_labels
        predicted_test_labels = hmm_prior_decode(self.train_outputs, self.test_outputs, self.true_train_labels,
                                                 transmat_labels, self.num_classes)
        return predicted_test_labels


def hmm_prior_decode(train_outputs, test_outputs, true_train_labels, transmat_labels, num_classes=None):
    # 构建 HMM 模型
    hmm_model = hmm.CategoricalHMM(n_components=num_classes)
    # # HMM模型初始化
    # # 1. 获得初始概率分布
    initial_state_probabilities = calculate_initial_state_probabilities(true_train_labels, num_states=num_classes)
    hmm_model.startprob_ = initial_state_probabilities
    # 2. 获得转移概率分布
    state_transition_probabilities = calculate_state_transition_probabilities(transmat_labels, num_states=num_classes)
    hmm_model.transmat_ = state_transition_probabilities
    # print(state_transition_probabilities)
    # 3. 获得发射概率p(x|y)
    emission_prob = estimate_emission_prob(train_outputs, true_train_labels)
    # p_x_given_y = convert_posterior_to_emission_probabilities(p_y_given_x=train_outputs, p_x=1 / num_classes,
    #                                                           p_y=initial_state_probabilities)
    # p_x_given_y = nn.Softmax(dim=1)(torch.tensor(p_x_given_y)).numpy()
    # emission_prob = estimate_emission_prob(p_x_given_y, true_train_labels)
    hmm_model.emissionprob_ = emission_prob

    seen = np.argmax(test_outputs, axis=1).reshape(-1, 1)
    logprob, predicted_test_labels = hmm_model.decode(seen, algorithm="viterbi")

    return predicted_test_labels


# 定义HMM的初始概率: 训练集中每种步态相位标签占总标签的比例
def calculate_initial_state_probabilities(labels, num_states=None):
    """

    :param labels: 训练集总的步态相位标签，十进制
    :param num_states: 步态相位的类别数
    :return: initial_state_probabilities, HMM的初始概率
    """
    num_labels = len(labels)
    if num_states is None:
        num_classes = np.unique(labels)
    initial_state_probabilities = np.zeros(num_states)

    for state in range(num_states):
        # state_count = labels.count(state)
        state_count = np.count_nonzero(labels == state)
        initial_state_probabilities[state] = state_count / num_labels

    return initial_state_probabilities


# 定义HMM的转移概率矩阵
def calculate_state_transition_probabilities(labels, num_states=None):
    """

    :param labels: 训练集总的步态相位标签，十进制
    :param num_states: 步态相位的类别数
    :return: state_transition_probabilities, HMM的转移概率
    """
    if num_states is None:
        num_classes = np.unique(labels)
    state_transition_probabilities = np.zeros((num_states, num_states))
    total_labels = len(labels)

    for i in range(total_labels - 1):
        current_state = labels[i]
        next_state = labels[i + 1]
        state_transition_probabilities[current_state][next_state] += 1

    # 归一化状态转移概率
    for i in range(num_states):
        transition_sum = np.sum(state_transition_probabilities[i])
        state_transition_probabilities[i] /= transition_sum

    return state_transition_probabilities


def convert_posterior_to_emission_probabilities(p_y_given_x, p_x, p_y):
    # p_y_given_x: 后验概率 p(y|x)
    # p_x: 观察 x 的先验概率
    # p_y: 状态 y 的先验概率

    # 使用贝叶斯规则计算发射概率 p(x|y)
    p_x_given_y = (p_y_given_x * p_x) / p_y

    return p_x_given_y


def estimate_emission_prob(softmax_output, labels):
    """
    估计发射概率矩阵的函数。

    参数：
    - softmax_output：一个 m*n 的数组，包含了 m 个样本的 softmax 输出，n 表示分类类别数。
    - labels：一个一维数组，包含了每个样本的真实步态类别标签。

    返回值：
    - emission_prob：估计的发射概率矩阵，形状为 (num_states, num_states)。
    """

    num_samples, num_classes = softmax_output.shape
    num_states = num_classes
    # 初始化发射概率矩阵为零矩阵
    emission_prob = np.zeros((num_states, num_classes))

    # 遍历每个步态类别
    for state in range(num_states):
        # 从训练数据中筛选出当前步态类别的样本索引
        state_samples_indices = np.where(labels == state)[0]

        # 筛选出当前步态类别的 softmax 输出
        state_samples_softmax = softmax_output[state_samples_indices]

        # 计算当前步态类别下的 softmax 输出均值
        if len(state_samples_softmax) > 0:
            state_mean = np.mean(state_samples_softmax, axis=0)
            emission_prob[state] = state_mean

    return emission_prob
