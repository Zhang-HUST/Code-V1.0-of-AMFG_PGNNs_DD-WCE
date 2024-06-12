import math
import numpy as np
from scipy.signal import butter, lfilter, iirfilter
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from utils.common_utils import all_elements_equal_to_str, is_nan_in_df_rows, analyze_list, get_middle_value_in_list

### 通用工具
"""emg滤波器：陷波滤波、带通滤波、低通滤波"""


class emg_filtering():
    def __init__(self, fs, lowcut, highcut, imf_band, imf_freq):
        self.fs = fs
        # butterWorth带通滤波器
        self.lowcut, self.highcut = lowcut, highcut
        # 50 Hz陷波滤波器
        self.imf_band, self.imf_freq = imf_band, imf_freq
        # 低通滤波
        self.cutoff = 20

    def Implement_Notch_Filter(self, data, order=2, filter_type='butter'):
        # Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered
        fs = self.fs
        nyq = fs / 2.0
        freq, band = self.imf_freq, self.imf_band
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)

        return filtered_data

    def butter_bandpass(self, order=6):
        lowcut, highcut, fs = self.lowcut, self.highcut, self.fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        y = lfilter(b, a, data)

        return y

    def butter_lowpass(self, order=5):
        cutoff, fs = self.cutoff, self.fs
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return b, a

    def butter_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        y = lfilter(b, a, data)

        return y


"""多模态多通道数据归一化方法，其中支持归一化方法：'min-max'、'max-abs'、'positive_negative_one；归一化层面：'matrix'、'rows'"""


def data_nomalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            # 实例化 MinMaxScaler 并设置归一化范围为 [0, 1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            # 实例化 MinMaxScaler 并设置归一化范围为 [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            # print(np.min(scaler.data_min_),np.max(scaler.data_max_))
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            # 实例化 MaxAbsScaler，并拟合数据以计算每列的最大值和最小值的绝对值
            scaler = MaxAbsScaler()
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 将数据整体缩放到 [-1, 1] 范围内
            normalized_data = (data / np.maximum(np.abs(np.max(scaler.data_max_)),
                                                 np.abs(np.min(scaler.data_min_)))) * scaler.scale_
        else:
            print('Error: 未识别的normalize_method！')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            print('Error: 未识别的normalize_method！')
    else:
        print('Error: 未识别的normalize_level！')

    return normalized_data


### 对运动模式分类任务

"""基于滑动重叠窗口采样的样本集分割：重叠窗长window、步进长度step"""


def overlapping_windowing_movement_classification(emg_data_act, angle_data_act, movement, window, step):
    length = math.floor((np.array(emg_data_act).shape[0] - window) / step)
    emg_sample, angle_sample, status_label, movement_label = [], [], [], []
    for j in range(length):
        sub_emg_sample = emg_data_act[step * j:(window + step * j), :]
        sub_angle_sample = angle_data_act[step * j:(window + step * j), :]
        emg_sample.append(sub_emg_sample)
        angle_sample.append(sub_angle_sample)
        movement_label.append(movement)

    return np.array(emg_sample), np.array(angle_sample), np.array(movement_label)


"""活动段提取和重叠窗口分割"""


def movement_classification_sample_segmentation(movement, emg_data_pre, angle_data_pre, status_label, window, step):
    ## label预处理: 对'STDUP'和'SITDN'，检测活动时的信号，返回 从 "A" 变为 "R" 和从 "R" 变为 "A" 时的数据索引和处理后的数据。对'WAK', 'UPS', 'DNS'，无需操作
    if movement == 'WAK' or movement == 'UPS' or movement == 'DNS':
        emg_data_act, angle_data_act, status_label_act = emg_data_pre, angle_data_pre, status_label
        print('       运动类型为: ', movement, '，无需处理活动段...')
        print('       进行重叠窗分割...')
        emg_sample, angle_sample, movement_label = overlapping_windowing_movement_classification(emg_data_act,
                                                                                                 angle_data_act,
                                                                                                 movement, window,
                                                                                                 step)
        print('       emg_sample.shape: ', emg_sample.shape, ', angle_sample.shape: ', angle_sample.shape,
              ', movement_label.shape: ', movement_label.shape)
        # print(np.array(time_act).shape, np.array(emg_data_act).shape, np.array(angle_data_act).shape, np.array(status_label_act).shape, np.array(group_label_act).shape)
    elif movement == 'STDUP' or movement == 'SITDN':
        print('       运动类型为: ', movement, '，开始处理活动段...')
        indices_a2r = [i for i in range(len(status_label) - 1) if status_label[i] == 'A' and status_label[i + 1] == 'R']
        indices_r2a = [i for i in range(len(status_label) - 1) if status_label[i] == 'R' and status_label[i + 1] == 'A']
        # print( indices_a2r,indices_r2a)
        emg_data_act, angle_data_act, status_label_act = [], [], []
        emg_sample, angle_sample, movement_label = [], [], []
        for i in range(min(len(indices_a2r), len(indices_r2a))):
            # time_act.extend(time[indices_r2a[i] + 1:indices_a2r[i] + 1, :])
            status_label_act.extend(status_label[indices_r2a[i] + 1:indices_a2r[i] + 1, :])
            # group_label_act.extend(group_label[indices_r2a[i] + 1:indices_a2r[i] + 1, :])
            emg_data_act = emg_data_pre[indices_r2a[i] + 1:indices_a2r[i] + 1, :]
            angle_data_act = angle_data_pre[indices_r2a[i] + 1:indices_a2r[i] + 1, :]
            emg_sample_temp, angle_sample_temp, movement_label_temp = overlapping_windowing_movement_classification(
                emg_data_act, angle_data_act, movement, window, step)
            emg_sample.extend(emg_sample_temp)
            angle_sample.extend(angle_sample_temp)
            movement_label.extend(movement_label_temp)
        ## 判断status_label_act中是否所有元素都等于 'A'
        if all_elements_equal_to_str(status_label_act, 'A'):
            print("       活动段提取和样本分割完毕，status_label_act中所有元素都等于 'A'")
        else:
            print("       活动段提取和样本分割完毕，status_label_act中存在不等于 'A' 的元素")
        emg_sample, angle_sample, movement_label = np.array(emg_sample), np.array(angle_sample), np.array(
            movement_label)
        print('       emg_sample.shape: ', emg_sample.shape, ', angle_sample.shape: ', angle_sample.shape,
              ', movement_label.shape: ', movement_label.shape)
    else:
        print('       未指定的运动类型！')

    return emg_sample, angle_sample, movement_label


### 对步态相位识别任务

"""以Status_label为基础，记录df_label中所有 NaN 值的行索引并删除"""


def delete_row_with_nan(df_data, df_label, rows=['Status'], nan_index='Status'):
    if is_nan_in_df_rows(df_label, rows=rows):
        ## 记录df_label中所有 NaN 值的行索引
        nan_index_list = df_label[df_label[nan_index].isna()].index.tolist()
        df_data_pre = df_data.drop(nan_index_list)
        df_label_pre = df_label.drop(nan_index_list)
        print('       Status列中含有NAN值，已处理！')
    else:
        df_data_pre, df_label_pre = df_data, df_label
        print('       Status列中没有NAN值，无需处理！')

    return df_data_pre, df_label_pre


"""基于滑动重叠窗口采样的样本集分割：重叠窗长window、步进长度step"""


def overlapping_windowing_gait_classification(emg_data, angle_data, gait_label_raw, group_label_raw, window, step):
    length = math.floor((np.array(emg_data).shape[0] - window) / step)
    emg_sample, angle_sample, gait_label, group_label = [], [], [], []
    for j in range(length):
        sub_emg_sample = emg_data[step * j:(window + step * j), :]
        sub_angle_sample = angle_data[step * j:(window + step * j), :]
        sub_gait_label = gait_label_raw[step * j:(window + step * j), :]
        emg_sample.append(sub_emg_sample)
        angle_sample.append(sub_angle_sample)
        group_label.append(group_label_raw)
        ## 对gait_label，标签为最中间位置的元素的值
        gait_label_lst = sub_gait_label.astype(int).flatten().tolist()  # 使用 flatten() 方法将多维数组转换为一维数组，再转换为列表
        gait_label.append(get_middle_value_in_list(gait_label_lst))

    return np.array(emg_sample), np.array(angle_sample), np.array(gait_label), np.array(group_label)


"""以group_label为基础的重叠窗口分割"""


def gait_classification_sample_segmentation(emg_data_pre, angle_raw_data, gait_label, group_label, window, step):
    ## label预处理:
    # 对'WAK','UPS'和'DNS'，group_label从1变化到10。
    # 对每个group，分别进行样本集分割，再拼接

    ## 分析group_label，返回counts中每个元素的出现次数(counts)并且记录counts中元素开始变化时的位置(indices)
    # print(group_label.shape)
    group_label_lst = group_label.astype(int).flatten().tolist()  # 使用 flatten() 方法将多维数组转换为一维数组，再转换为列表
    counts, indices = analyze_list(group_label_lst)
    ## 生成新的indices [0, 原indices, len(indices)]
    indices.insert(0, 0)
    indices.append(len(group_label))
    emg_sample, angle_sample, gait_label_raw, group_label_raw = [], [], [], []
    for i in range(len(counts)):
        count = int(list(counts.keys())[i])  # 1, 2, ..., 10
        emg_data_count = emg_data_pre[indices[i]: indices[i + 1], :]
        angle_data_count = angle_raw_data[indices[i]: indices[i + 1], :]
        gait_label_count = gait_label[indices[i]: indices[i + 1]]
        group_label_count = count
        emg_sample_temp, angle_sample_temp, gait_label_temp, group_label_temp = overlapping_windowing_gait_classification(
            emg_data_count,
            angle_data_count, gait_label_count, group_label_count, window, step)
        emg_sample.extend(emg_sample_temp)
        angle_sample.extend(angle_sample_temp)
        gait_label_raw.extend(gait_label_temp)
        group_label_raw.extend(group_label_temp)

    emg_sample, angle_sample, gait_label_raw, group_label_raw = np.array(emg_sample), np.array(angle_sample), np.array(
        gait_label_raw), np.array(group_label_raw)
    print('       emg_sample.shape: ', emg_sample.shape, ', angle_sample.shape: ', angle_sample.shape,
          ', gait_label.shape: ', gait_label_raw.shape, ', group_label.shape: ', group_label_raw.shape)

    return emg_sample, angle_sample, gait_label_raw, group_label_raw
