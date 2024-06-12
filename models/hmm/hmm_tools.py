# import torch
# import numpy as np
# import torch.nn as nn
# from hmmlearn import hmm
# from sklearn.preprocessing import LabelEncoder
#
#
# class HMMGaitClassification:
#     def __init__(self, model, train_loader, test_loader, method, hmm_type, num_classes=None, n_iter=100):
#         self.model = model.to(device='cpu')
#         self.method, self.hmm_type = method, hmm_type
#         self.n_iter = n_iter
#         self.train_outputs, self.true_train_labels = self.get_feature_labels_from_dataloader(train_loader)
#         self.test_outputs, _ = self.get_feature_labels_from_dataloader(test_loader)
#         if num_classes is not None:
#             self.num_classes = num_classes
#         else:
#             self.num_classes = np.unique(self.true_train_labels)
#         self.init()
#
#     def get_feature_labels_from_dataloader(self, data_loader):
#         deep_features_list = []
#         label_list = []
#         for data, feature, label in data_loader:
#             deep_feature = self.model(data, feature)
#             deep_feature = nn.Softmax()(deep_feature)
#             deep_features_list.append(deep_feature)
#             label_list.append(label)
#         deep_features = torch.cat(deep_features_list, dim=0).detach().numpy()
#         labels = torch.cat(label_list, dim=0).view(-1).numpy()
#
#         return deep_features, labels
#
#     def init(self):
#         predicted_test_labels = []
#         if self.method == 'auto':
#             predicted_test_labels = hmm_auto(self.train_outputs, self.test_outputs, self.true_train_labels,
#                                                 self.hmm_type, self.num_classes, self.n_iter)
#         elif self.method == 'decode':
#             predicted_test_labels = hmm_prior_decode(self.train_outputs, self.test_outputs, self.true_train_labels,
#                                                         self.hmm_type, self.num_classes, self.n_iter)
#         elif self.method == 'score':
#             predicted_test_labels = hmm_prior_score(self.train_outputs, self.test_outputs, self.true_train_labels,
#                                                        self.hmm_type, self.num_classes, self.n_iter)
#         else:
#             raise ValueError('method must be auto, decode or score')
#         return predicted_test_labels
#
#
# def hmm_auto(train_outputs, test_outputs, true_train_labels, hmm_type, num_classes=None, n_iter=100):
#     """
#      HMM 使用的数据是观测序列和对应的隐藏状态序列，用于学习模型的参数（如初始概率、转移概率和发射概率）
#     :param train_outputs: 分类器在训练集的输出，Tensor: num*num_classes，
#     :param test_outputs: 分类器在测试集的输出，Tensor: num*num_classes，
#     :param hmm_type: hmm算法的类型， ‘MultinomialHMM’/'GaussianHMM'，
#     :param num_classes: 类别数
#     :param n_iter: HMM算法迭代的次数
#     :return: predicted_train_labels, predicted_test_labels: HMM推测的隐藏状态序列，即预测的步态相位，Labels for each sample from ``X``.
#     """
#     # 将模型输出进行softmax激活作为观测序列
#     # train_observations = output2softmax(train_outputs)
#     # test_observations = output2softmax(test_outputs)
#     train_observations = train_outputs
#     test_observations = test_outputs
#
#     # 构建 HMM 模型
#     if hmm_type == 'MultinomialHMM':
#         hmm_model = hmm.MultinomialHMM(n_components=num_classes, n_iter=n_iter, algorithm="viterbi", verbose=False)
#         binary_output = np.zeros_like(train_outputs)
#         binary_output[np.arange(len(train_outputs)), train_outputs.argmax(axis=1)] = 1
#         train_observations = binary_output.astype('int')
#         binary_output1 = np.zeros_like(test_outputs)
#         binary_output1[np.arange(len(test_outputs)), test_outputs.argmax(axis=1)] = 1
#         test_observations = binary_output1.astype('int')
#     elif hmm_type == 'GaussianHMM':
#         hmm_model = hmm.GaussianHMM(n_components=num_classes, n_iter=n_iter, covariance_type="spherical",
#                                     algorithm="viterbi", verbose=False)
#     else:
#         raise Exception('hmm_type wrong ! support MultinomialHMM and GaussianHMM only!')
#         # covariance_type - 要使用的协方差参数的类型: *“spherical”——每个状态使用一个适用于所有特征的单一方差值(默认)。 * “diag”——每个州使用对角线
#
#     # 使用观测序列和隐藏状态序列（如果有）对 HMM 模型进行训练：
#     # 1. 使用 Baum-Welch 算法（前向-后向算法）来估计模型的参数。该算法通过最大化似然函数来调整模型参数，使得观测序列的概率最大化。
#     # 2. 使用维特比算法来根据观测序列找到最可能的隐藏状态序列。维特比算法会考虑模型的参数和观测序列，找到最大概率路径，即最可能的隐藏状态序列
#
#     hmm_model.fit(train_observations, lengths=len(train_observations))
#     # 预测步态相位
#     # train_log_likelihoods = hmm_model.score(train_observations)
#     # test_log_likelihoods = hmm_model.score(test_observations)
#     # predicted_train_labels = hmm_model.decode(train_observations)[1]
#     predicted_test_labels = hmm_model.decode(test_observations)[1]
#
#     return predicted_test_labels
#
#
# def hmm_prior_decode(train_outputs, test_outputs, true_train_labels, hmm_type, num_classes=None, n_iter=100):
#     """
#      构建隐马尔可夫模型所需要的三要素， 即初始概率分布、转移概率分布、发射概率分布
#     :param train_outputs: 分类器在训练集的输出，Tensor: num*num_classes，
#     :param test_outputs: 分类器在测试集的输出，Tensor: num*num_classes，
#     :param hmm_type: hmm算法的类型， ‘MultinomialHMM’/'GaussianHMM'，
#     :param true_train_labels: 训练集的真实相位标签，Numpy: num（十进制），
#     :param n_iter: HMM算法迭代的次数
#     :param num_classes: 类别数
#     :return: predicted_train_labels, predicted_test_labels: HMM推测的隐藏状态序列，即预测的步态相位，Labels for each sample from ``X``.``.
#     """
#     # 将模型输出进行softmax激活作为观测序列
#     # train_observations = output2softmax(train_outputs)
#     # test_observations = output2softmax(test_outputs)
#     train_observations = train_outputs
#     test_observations = test_outputs
#     # 构建 HMM 模型
#     if hmm_type == 'MultinomialHMM':
#         hmm_model = hmm.MultinomialHMM(n_components=num_classes, n_iter=n_iter, algorithm="viterbi", verbose=False)
#     elif hmm_type == 'GaussianHMM':
#         hmm_model = hmm.GaussianHMM(n_components=num_classes, n_iter=n_iter, covariance_type="spherical",
#                                     algorithm="viterbi", verbose=False)
#     else:
#         raise Exception('hmm_type wrong ! support MultinomialHMM and GaussianHMM only!')
#         # covariance_type - 要使用的协方差参数的类型: *“spherical”——每个状态使用一个适用于所有特征的单一方差值(默认)。 * “diag”——每个州使用对角线
#     # HMM模型初始化
#     # 1. 获得初始概率分布
#     initial_state_probabilities = calculate_initial_state_probabilities(true_train_labels, num_states=num_classes)
#     hmm_model.startprob_ = initial_state_probabilities
#     # 2. 获得转移概率分布
#     state_transition_probabilities = calculate_state_transition_probabilities(true_train_labels, num_states=num_classes)
#     hmm_model.transmat_ = state_transition_probabilities
#     # 3. 获得发射概率
#     train_emission_prob = calculate_emission_probabilities(train_observations)
#     test_emission_prob = calculate_emission_probabilities(test_observations)
#     hmm_model.emissionprob_ = train_emission_prob
#     # # 4. 训练 HMM 模型
#     # hmm_model.fit(train_observations)
#     # 5. 预测步态相位
#     # train_log_likelihoods = hmm_model.decode(train_emission_prob, algorithm="viterbi")
#     test_log_likelihoods = hmm_model.decode(test_emission_prob, algorithm="viterbi")
#
#     # predicted_train_labels = np.argmax(train_log_likelihoods)
#     predicted_test_labels = np.argmax(test_log_likelihoods)
#
#     return predicted_test_labels
#
#
# def hmm_prior_score(train_outputs, test_outputs, true_train_labels, hmm_type, num_classes=None, n_iter=100):
#     """
#      构建隐马尔可夫模型所需要的三要素， 即初始概率分布、转移概率分布、发射概率分布
#     :param train_outputs: 分类器在训练集的输出，Tensor: num*num_classes，
#     :param test_outputs: 分类器在测试集的输出，Tensor: num*num_classes，
#     :param hmm_type: hmm算法的类型， ‘MultinomialHMM’/'GaussianHMM'，
#     :param true_train_labels: 训练集的真实相位标签，Numpy: num（十进制），
#     :param n_iter: HMM算法迭代的次数
#     :param num_classes: 类别数
#     :return: predicted_train_labels, predicted_test_labels: HMM推测的隐藏状态序列，即预测的步态相位，Labels for each sample from ``X``.``.
#     """
#     # 将模型输出进行softmax激活作为观测序列
#     # train_observations = output2softmax(train_outputs)
#     # test_observations = output2softmax(test_outputs)
#     train_observations = train_outputs
#     test_observations = test_outputs
#     # 构建 HMM 模型
#     if hmm_type == 'MultinomialHMM':
#         hmm_model = hmm.MultinomialHMM(n_components=num_classes, n_iter=n_iter, algorithm="viterbi", verbose=False)
#     elif hmm_type == 'GaussianHMM':
#         hmm_model = hmm.GaussianHMM(n_components=num_classes, n_iter=n_iter, covariance_type="spherical",
#                                     algorithm="viterbi", verbose=False)
#     else:
#         raise Exception('hmm_type wrong ! support MultinomialHMM and GaussianHMM only!')
#         # covariance_type - 要使用的协方差参数的类型: *“spherical”——每个状态使用一个适用于所有特征的单一方差值(默认)。 * “diag”——每个州使用对角线
#     # HMM模型初始化
#     # 1. 获得初始概率分布
#     initial_state_probabilities = calculate_initial_state_probabilities(true_train_labels, num_states=num_classes)
#     hmm_model.startprob_ = initial_state_probabilities
#     # 2. 获得转移概率分布
#     state_transition_probabilities = calculate_state_transition_probabilities(true_train_labels, num_states=num_classes)
#     hmm_model.transmat_ = state_transition_probabilities
#
#     # 3. 训练 HMM 模型
#     hmm_model.fit(train_observations)
#
#     # 4. 预测步态相位
#     # train_log_likelihoods = hmm_model.score(train_observations)
#     # test_log_likelihoods = hmm_model.score(test_observations)
#     # print(test_log_likelihoods)
#     # predicted_train_labels = np.argmax(train_log_likelihoods, axis=0)
#     # predicted_test_labels = np.argmax(test_log_likelihoods, axis=0)
#     # print(predicted_test_labels.shape)
#     # predicted_train_labels = hmm_model.predict(train_observations)
#     predicted_test_labels = hmm_model.predict(test_observations)
#     # print(predicted_test_labels)
#     # predicted_test_labels = np.argmax(predicted_test_labels)
#     # print(predicted_test_labels)
#     return predicted_test_labels
#
#
# # 定义HMM的初始概率: 训练集中每种步态相位标签占总标签的比例
# def calculate_initial_state_probabilities(labels, num_states=None):
#     """
#
#     :param labels: 训练集总的步态相位标签，十进制
#     :param num_states: 步态相位的类别数
#     :return: initial_state_probabilities, HMM的初始概率
#     """
#     num_labels = len(labels)
#     if num_states is None:
#         num_classes = np.unique(labels)
#     initial_state_probabilities = np.zeros(num_states)
#
#     for state in range(num_states):
#         # state_count = labels.count(state)
#         state_count = np.count_nonzero(labels == state)
#         initial_state_probabilities[state] = state_count / num_labels
#
#     return initial_state_probabilities
#
#
# # 定义HMM的转移概率矩阵
# def calculate_state_transition_probabilities(labels, num_states=None):
#     """
#
#     :param labels: 训练集总的步态相位标签，十进制
#     :param num_states: 步态相位的类别数
#     :return: state_transition_probabilities, HMM的转移概率
#     """
#     if num_states is None:
#         num_classes = np.unique(labels)
#     state_transition_probabilities = np.zeros((num_states, num_states))
#     total_labels = len(labels)
#
#     for i in range(total_labels - 1):
#         current_state = labels[i]
#         next_state = labels[i + 1]
#         state_transition_probabilities[current_state][next_state] += 1
#
#     # 归一化状态转移概率
#     for i in range(num_states):
#         transition_sum = np.sum(state_transition_probabilities[i])
#         state_transition_probabilities[i] /= transition_sum
#
#     return state_transition_probabilities
#
#
# # 计算HMM的发射概率
# def calculate_emission_probabilities(output):
#     """
#
#     :param output: 模型输出概率，Tensor: num*num_classes，
#     :return: emission_prob, HMM的发射概率
#     """
#     # softmax_output = nn.Softmax(dim=1)(output).detach().to(device='cpu').numpy()
#     # 按行对矩阵进行归一化处理
#     emission_prob = output / np.sum(output, axis=1, keepdims=True)
#     return emission_prob
#
#
# # 模型输出经过softmax激活，并转为numpy
# def output2softmax(output):
#     """
#
#     :param output: 模型输出概率，Tensor: num*num_classes，
#     :return: softmax_output, softmax输出
#     """
#     softmax_output = nn.Softmax(dim=1)(output).detach().to(device='cpu').numpy()
#
#     return softmax_output
