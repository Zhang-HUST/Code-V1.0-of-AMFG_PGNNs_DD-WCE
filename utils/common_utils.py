import os
import datetime
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

"""控制台打印字符串。time和line_break：布尔量，是否显示时间和换行"""


def printlog(info, time=True, line_break=True):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if time:
        print("\n" + "==========" * 8 + "%s" % nowtime)
    else:
        pass
    if line_break:
        print(info + '...\n')
    else:
        print(info)


"""判断一个列表lst中所有元素是否都等于str，以确认活动段提取是否正确"""


def all_elements_equal_to_str(lst, str):
    return all(element == str for element in lst)


"""判断一个列表中是否存在某个字符串，for 特征提取"""


def is_string_in_list(lst, target_string):
    return target_string in lst


""" 获取特征列表的名称，单个特征的名称格式为：'channelName_featureName' """


def get_feature_list(channels, feature_type, concatenation=False):
    all_fea_names = []
    for i in range(len(channels)):
        channel = channels[i]
        if concatenation:
            # concatenation = False, 返回的特征列表维度为[1，len(channels)*len(feature_type)]
            all_fea_names.extend([''.join([channel, '_', feature_type[i]]) for i in range(len(feature_type))])
        else:
            # concatenation = True, 返回的特征列表维度为[len(channels), len(feature_type)]
            all_fea_names.append([''.join([channel, '_', feature_type[i]]) for i in range(len(feature_type))])

    return np.array(all_fea_names)


"""判断是否存在dir，如果不存在，则新建"""


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass


""" 判断 df 的 某一列或多列是否存在 NaN 值 """


def is_nan_in_df_rows(df, rows=[]):
    has_nan = None
    if len(rows) == 0:
        print('Error: 未指定列名！')
    elif len(rows) == 1:
        ## 使用isna()方法来检查是否存在NaN值，然后使用any()方法来判断结果是否为True
        has_nan = df[rows[0]].isna().any()
    elif len(rows) > 1:
        ## 使用 isna() 方法来检查是否存在 NaN 值，然后使用两次 any() 方法。
        ## 第一次 any() 方法在列维度上检查是否存在至少一个 True 值，如果存在，则返回 True。
        ## 第二次 any() 方法在行维度上检查是否存在至少一个 True 值，如果存在，则返回 True。因此，如果任何一列中存在 NaN 值，最终结果会返回 True。
        has_nan = df[rows].isna().any().any()
    return has_nan


"""统计list中每个元素的出现次数并且记录list中元素开始变化时的位置"""


def analyze_list(lst):
    count_dict = {}  # 用于统计元素出现次数的字典
    index_list = []  # 用于记录元素开始变化时的行索引

    for i, elem in enumerate(lst):
        count_dict[elem] = count_dict.get(elem, 0) + 1

        if i > 0 and lst[i] != lst[i - 1]:
            index_list.append(i)

    return count_dict, index_list


"""last_name=[a] element_name=[b], output=[[a], [b]]"""


def add_elements_in_list(list_name, element_name):
    # example2: last_name=[] element_name=[a], output=[[a]]
    # example1: last_name=[a] element_name=[b], output=[[a], [b]]
    if len(list_name):
        list_name = list_name + element_name
    else:
        list_name = element_name

    return list_name


"""给定列表lst，获取最中间位置的元素的值"""


def get_middle_value_in_list(lst):
    length = len(lst)  # 获取列表长度
    if length % 2 == 1:
        middle_index = length // 2
        middle_value = lst[middle_index]
    else:
        middle_index = length // 2 - 1
        middle_value = lst[middle_index]
    return middle_value


def convert_to_2d(arr):
    arr = np.array(arr)
    if arr.ndim == 1:  # 判断数组是否为一维数组
        arr = arr.reshape(-1, 1)  # 将一维数组转换为二维数组
    else:
        pass

    return arr


def is_label_onehot(y_true, y_pre):
    ## 判断y_true和y_pre是否为一维数组，如果是，转为二维
    y_true, y_pre = convert_to_2d(y_true), convert_to_2d(y_pre)
    if y_true.shape[1] == 1 or y_pre.shape[1] == 1:
        return False
    else:
        return True


def onehot2decimalism(arr, keep_dims=False):
    return np.argmax(arr, axis=1, keepdims=keep_dims)


def calculate_class_weights_torch(labels, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(labels))
    class_counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)

    return class_weights


def calculate_class_weights(labels, num_classes=None, verbose=True):
    if num_classes is None:
        classes = np.unique(labels)
    else:
        classes = np.arange(num_classes)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    if verbose:
        dict_class_weights = dict(enumerate(class_weights))
        for i, j in dict_class_weights.items():
            print('类别', str(i), '不平衡率为:', str(j))
    return class_weights


def calculate_samples_per_class(labels, verbose=True):
    unique_elements, element_counts = np.unique(labels, return_counts=True)
    if verbose:
        for i, j in zip(unique_elements, element_counts):
            print('类别', str(i), '样本数为:', str(j))
    return element_counts


def get_num_classes(gait_or_motion, motion_type):
    classes = {'motion': 5, 'gait': {'WAK': 5, 'UPS': 3, 'DNS': 2}}
    if gait_or_motion == 'motion':
        num_classes = classes[gait_or_motion]
    elif gait_or_motion == 'gait':
        num_classes = classes[gait_or_motion][motion_type]
    else:
        raise ValueError('Please input correct gait_or_motion and motion_type')

    return num_classes
