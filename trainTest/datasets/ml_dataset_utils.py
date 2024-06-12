import numpy as np
# from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit


def get_intra_ml_datasets(file_name, feature_list, encoded_label_name, total_exp_time, current_exp_time, test_ratio):
    with open(file_name, 'rb') as f:
        sub_emg_features = np.load(f)[feature_list[0]]
        sub_angle_features = np.load(f)[feature_list[1]]
        sub_gait_label_encoded = np.load(f)[encoded_label_name]
    # emg_features.shape:  (n, 9, 6) , angle_features.shape:  (n, 6, 6)
    # features.shape:  (n, 15*6)
    features = np.concatenate([sub_emg_features, sub_angle_features], axis=1)
    features = features.reshape(features.shape[0], -1)
    labels = sub_gait_label_encoded
    print('features.shape: ', features.shape, ', labels.shape: ', labels.shape)
    # 使用sklearn将数据集按照比例8：2划分成train：test。
    all_x_train, all_x_test, all_y_train, all_y_test = [], [], [], []
    sss = StratifiedShuffleSplit(n_splits=total_exp_time, test_size=test_ratio, random_state=42)
    # 分层shuffle - plit交叉验证器。提供训练/测试索引以在训练/测试集中划分数据。
    # 这个交叉验证对象是StratifiedKFold和ShuffleSplit的合并，后者返回分层的随机折叠。折叠是通过保留每个类别的样本百分比来实现的。

    for train_index, test_index in sss.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        all_x_train.append(X_train)
        all_x_test.append(X_test)
        all_y_train.append(Y_train)
        all_y_test.append(Y_test)
    x_train, y_train = all_x_train[current_exp_time - 1], all_y_train[current_exp_time - 1]
    x_test, y_test = all_x_test[current_exp_time - 1], all_y_test[current_exp_time - 1]
    print('x_train.shape: ', x_train.shape, ', x_test.shape: ', x_test.shape)
    print('y_train.shape: ', y_train.shape, ', y_test.shape: ', y_test.shape)

    return x_train, y_train, x_test, y_test
