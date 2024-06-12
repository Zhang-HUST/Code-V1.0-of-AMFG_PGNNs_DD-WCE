from utils.common_utils import get_feature_list, is_string_in_list, add_elements_in_list
from utils.processing_tools.processing import data_nomalize
from utils.feature_extraction_tools.feature_extraction_utils import *

"""主函数1：多个样本多个通道的emg和运动学数据的特征提取"""


def emg_kinematic_feature_extraction(emg_sample, angle_sample, emg_channels, angle_channels, emg_feature_type,
                                     angle_feature_type, fea_normalize_method, fea_normalize_level):
    """
    1. 支持的emg特征列表
        1.1. 15个时域特征['方差VAR', '均方根值RMS', '肌电积分值IEMG', '绝对值均值MAV', '对数探测器LOG', '波形长度WL', '平均振幅变化AAC','差值绝对标准差值DASDV',
                        '过零率ZC', 'Willison幅值WAMP', '脉冲百分率MYOP', 斜率符号变化SSC, 简单平方积分SSI, 峭度因子KF, 第三时间矩TM3]
        1.2. 9个频域特征['频数比FR', '平均功率MNP', '总功率TOP', '平均频率MNF', '中值频率MDF', '峰值频率PKF', '谱矩1SM1', '谱矩2SM2', '谱矩3SM3']
        1.3. 1个时频域特征['小波能量WENT']
        1.4. 3个信息熵特征['近似熵AE', '样本熵SE’, '模糊熵FE']
    2. 支持的运动学特征列表
                ['均方根值RMS', '平均值AVG', '最大值MAX', '最小值MIN','峰峰值PP', '标准差STD']
    """
    ## 1. 获取emg所有通道的特征列表的名称all_emg_fea_names、运动学数据所有通道的特征列表的名称all_kinematic_fea_names
    all_emg_fea_names = get_feature_list(emg_channels, emg_feature_type,
                                         concatenation=False)  # shape, num*len(emg_channels)*len(feature_type)
    all_kinematic_fea_names = get_feature_list(angle_channels, angle_feature_type,
                                               concatenation=False)  # shape, num*len(angel_channels)*len(feature_type)

    ## 2. emg特征提取
    all_emg_feas = []
    for i in range(emg_sample.shape[0]):
        temp1 = []
        for j in range(emg_sample.shape[2]):
            sub_emg_data = emg_sample[i, :, j]
            sub_emg_feas = emg_feature_extraction_alone(sub_emg_data, emg_feature_type)
            temp1.extend(np.array(sub_emg_feas))
        all_emg_feas.append(temp1)
    all_emg_feas = np.array(all_emg_feas)  # shape, num*[len(emg_channels)*len(feature_type)]

    ## 3. 运动学特征提取
    all_kinematic_feas = []
    for i in range(angle_sample.shape[0]):
        temp0 = []
        # for j in range(len(angle_channels)):
        for j in range(angle_sample.shape[2]):
            sub_kinematic_data = angle_sample[i, :, j]
            sub_kinematic_feas = kinematic_feature_extraction_alone(sub_kinematic_data, angle_feature_type)
            temp0.extend(np.array(sub_kinematic_feas))
        all_kinematic_feas.append(temp0)
    all_kinematic_feas = np.array(all_kinematic_feas)  # shape, num*[len(angle_channels)*len(feature_type)]

    ## 4. 特征集归一化（max-abs，按列，不是矩阵）
    emg_feas_normalize = data_nomalize(all_emg_feas, fea_normalize_method, fea_normalize_level)
    angle_feas_normalize = data_nomalize(all_kinematic_feas, fea_normalize_method, fea_normalize_level)

    ## 5. 将特征集排列为num*len(channels）*len(feature_type)
    all_emg_feas_pre = np.reshape(emg_feas_normalize,
                                  (emg_feas_normalize.shape[0], len(emg_channels), len(emg_feature_type)))
    all_angle_feas_pre = np.reshape(angle_feas_normalize,
                                    (angle_feas_normalize.shape[0], len(angle_channels), len(angle_feature_type)))
    print('       emg_feas.shape: ', all_emg_feas_pre.shape, ', angle_feas.shape: ', all_angle_feas_pre.shape)
    return all_emg_feas_pre, all_emg_fea_names, all_angle_feas_pre, all_kinematic_fea_names


"""主函数2：多个样本多个通道的emg特征提取"""


def emg_feature_extraction(emg_sample, emg_channels, emg_feature_type, fea_normalize_method, fea_normalize_level):
    """
    1. 支持的emg特征列表
        1.1. 15个时域特征['方差VAR', '均方根值RMS', '肌电积分值IEMG', '绝对值均值MAV', '对数探测器LOG', '波形长度WL', '平均振幅变化AAC','差值绝对标准差值DASDV',
                        '过零率ZC', 'Willison幅值WAMP', '脉冲百分率MYOP', 斜率符号变化SSC, 简单平方积分SSI, 峭度因子KF, 第三时间矩TM3]
        1.2. 9个频域特征['频数比FR', '平均功率MNP', '总功率TOP', '平均频率MNF', '中值频率MDF', '峰值频率PKF', '谱矩1SM1', '谱矩2SM2', '谱矩3SM3']
        1.3. 1个时频域特征['小波能量WENT']
        1.4. 3个信息熵特征['近似熵AE', '样本熵SE’, '模糊熵FE']
    """
    ## 1. 获取emg所有通道的特征列表的名称all_emg_fea_names
    all_emg_fea_names = get_feature_list(emg_channels, emg_feature_type,
                                         concatenation=False)  # shape, num*len(emg_channels)*len(feature_type)

    ## 2. emg特征提取
    all_emg_feas = []
    for i in range(emg_sample.shape[0]):
        temp1 = []
        for j in range(emg_sample.shape[2]):
            sub_emg_data = emg_sample[i, :, j]
            sub_emg_feas = emg_feature_extraction_alone(sub_emg_data, emg_feature_type)
            temp1.extend(np.array(sub_emg_feas))
        all_emg_feas.append(temp1)
    all_emg_feas = np.array(all_emg_feas)  # shape, num*[len(emg_channels)*len(feature_type)]

    ## 3. 特征集归一化（max-abs，按列，不是矩阵）
    emg_feas_normalize = data_nomalize(all_emg_feas, fea_normalize_method, fea_normalize_level)

    ## 4. 将特征集排列为num*len(channels）*len(feature_type)
    all_emg_feas_pre = np.reshape(emg_feas_normalize,
                                  (emg_feas_normalize.shape[0], len(emg_channels), len(emg_feature_type)))
    print('       emg_feas.shape: ', all_emg_feas_pre.shape)
    return all_emg_feas_pre, all_emg_fea_names


"""主函数3：多个样本多个通道的运动学数据的特征提取"""


def kinematic_feature_extraction(angle_sample, angle_channels, angle_feature_type, fea_normalize_method,
                                 fea_normalize_level):
    """
    1. 支持的运动学特征列表
                ['均方根值RMS', '平均值AVG', '最大值MAX', '最小值MIN','峰峰值PP', '标准差STD']
    """
    ## 1. 获取运动学数据所有通道的特征列表的名称all_kinematic_fea_names
    all_kinematic_fea_names = get_feature_list(angle_channels, angle_feature_type,
                                               concatenation=False)  # shape, num*len(angel_channels)*len(feature_type)

    ## 2. 运动学特征提取
    all_kinematic_feas = []
    for i in range(angle_sample.shape[0]):
        temp0 = []
        # for j in range(len(angle_channels)):
        for j in range(angle_sample.shape[2]):
            sub_kinematic_data = angle_sample[i, :, j]
            sub_kinematic_feas = kinematic_feature_extraction_alone(sub_kinematic_data, angle_feature_type)
            temp0.extend(np.array(sub_kinematic_feas))
        all_kinematic_feas.append(temp0)
    all_kinematic_feas = np.array(all_kinematic_feas)  # shape, num*[len(angle_channels)*len(feature_type)]

    ## 3. 特征集归一化（max-abs，按列，不是矩阵）
    angle_feas_normalize = data_nomalize(all_kinematic_feas, fea_normalize_method, fea_normalize_level)

    ## 4. 将特征集排列为num*len(channels）*len(feature_type)
    all_angle_feas_pre = np.reshape(angle_feas_normalize,
                                    (angle_feas_normalize.shape[0], len(angle_channels), len(angle_feature_type)))
    print('       angle_feas.shape: ', all_angle_feas_pre.shape)
    return all_angle_feas_pre, all_kinematic_fea_names


"""单个样本单个通道的运动学数据的特征提取"""


def kinematic_feature_extraction_alone(x, feature_type):
    """
    all_kinematic_features = ['RMS', 'AVG', 'MAX', 'MIN','PP', 'STD']
    """
    kinematic_feas = []
    if is_string_in_list(feature_type, 'RMS'):
        fea_rms = np.sqrt((np.mean(x ** 2)))
        kinematic_feas.append(fea_rms)
    if is_string_in_list(feature_type, 'AVG'):
        fea_avg = np.mean(x)
        kinematic_feas.append(fea_avg)
    if is_string_in_list(feature_type, 'MAX'):
        fea_max = np.max(x)
        kinematic_feas.append(fea_max)
    if is_string_in_list(feature_type, 'MIN'):
        fea_min = np.min(x)
        kinematic_feas.append(fea_min)
    if is_string_in_list(feature_type, 'PP'):
        fea_pp = 0.5 * (np.max(x) - np.min(x))
        kinematic_feas.append(fea_pp)
    if is_string_in_list(feature_type, 'STD'):
        fea_std = np.std(x, ddof=1)
        kinematic_feas.append(fea_std)

    return kinematic_feas


"""单个样本单个通道的emg数据的特征提取"""


def emg_feature_extraction_alone(x, feature_type):
    emg_feas = []
    th = np.mean(x) + 3 * np.std(x)
    ssc_threshold = 0.000001
    fs_global = 1920
    ## 16个时域特征
    if is_string_in_list(feature_type, 'VAR'):
        fea_var = np.var(x)
        emg_feas.append(fea_var)
    if is_string_in_list(feature_type, 'RMS'):
        fea_rms = np.sqrt(np.mean(x ** 2))
        emg_feas.append(fea_rms)
    if is_string_in_list(feature_type, 'IEMG'):
        fea_iemg = np.sum(abs(x))
        emg_feas.append(fea_iemg)
    if is_string_in_list(feature_type, 'MAV'):
        fea_mav = np.sum(np.absolute(x)) / len(x)
        emg_feas.append(fea_mav)
    if is_string_in_list(feature_type, 'LOG'):
        fea_log = np.exp(np.sum(np.log10(np.absolute(x))) / len(x))
        emg_feas.append(fea_log)
    if is_string_in_list(feature_type, 'WL'):
        fea_wl = np.sum(abs(np.diff(x)))
        emg_feas.append(fea_wl)
    if is_string_in_list(feature_type, 'AAC'):
        fea_aac = np.sum(abs(np.diff(x))) / len(x)
        emg_feas.append(fea_aac)
    if is_string_in_list(feature_type, 'DASDV'):
        fea_dasdv = math.sqrt((1 / (len(x) - 1)) * np.sum((np.diff(x)) ** 2))
        emg_feas.append(fea_dasdv)
    if is_string_in_list(feature_type, 'ZC'):
        fea_zc = get_emg_feature_zc(x, th)
        emg_feas.append(fea_zc)
    if is_string_in_list(feature_type, 'WAMP'):
        fea_wamp = get_emg_feature_wamp(x, th)
        emg_feas.append(fea_wamp)
    if is_string_in_list(feature_type, 'MYOP'):
        fea_myop = get_emg_feature_myop(x, th)
        emg_feas.append(fea_myop)
    if is_string_in_list(feature_type, 'SSC'):
        fea_ssc = get_emg_feature_ssc(x, threshold=ssc_threshold)
        emg_feas.append(fea_ssc)
    if is_string_in_list(feature_type, 'SSI'):
        fea_ssi = get_emg_feature_ssi(x)
        emg_feas.append(fea_ssi)
    if is_string_in_list(feature_type, 'KF'):
        fea_kf = get_emg_feature_kf(x)
        emg_feas.append(fea_kf)
    if is_string_in_list(feature_type, 'TM3'):
        fea_tm3 = get_emg_feature_tm3(x)
        emg_feas.append(fea_tm3)
    ## 9个频域特征
    frequency, power = get_signal_spectrum(x, fs_global)
    if is_string_in_list(feature_type, 'FR'):
        fea_fr = get_emg_feature_fr(frequency, power)  # Frequency ratio
        emg_feas.append(fea_fr)
    if is_string_in_list(feature_type, 'MNP'):
        fea_mnp = np.sum(power) / len(power)  # Mean power
        emg_feas.append(fea_mnp)
    if is_string_in_list(feature_type, 'TOP'):
        fea_top = np.sum(power)  # Total power
        emg_feas.append(fea_top)
    if is_string_in_list(feature_type, 'MNF'):
        fea_mnf = get_emg_feature_mnf(frequency, power)  # Mean frequency
        emg_feas.append(fea_mnf)
    if is_string_in_list(feature_type, 'MDF'):
        fea_mdf = get_emg_feature_mdf(frequency, power)  # Median frequency
        emg_feas.append(fea_mdf)
    if is_string_in_list(feature_type, 'PKF'):
        fea_pkf = frequency[power.argmax()]  # Peak frequency
        emg_feas.append(fea_pkf)
    if is_string_in_list(feature_type, 'SM1'):
        fea_sm1 = get_emg_feature_sm1(x, fs_global)  # Spectral Moment 1
        emg_feas.append(fea_sm1)
    if is_string_in_list(feature_type, 'SM2'):
        fea_sm2 = get_emg_feature_sm2(x, fs_global)  # Spectral Moment 2
        emg_feas.append(fea_sm2)
    if is_string_in_list(feature_type, 'SM3'):
        fea_sm3 = get_emg_feature_sm3(x, fs_global)  # Spectral Moment 3
        emg_feas.append(fea_sm3)
    ## 1个时频域特征
    if is_string_in_list(feature_type, 'WENT'):
        fea_went = get_emg_feature_went(x)  # Wavelet energy
        emg_feas.append(fea_went)
    ## 3个信息熵特征
    if is_string_in_list(feature_type, 'AE'):
        fea_ae = get_emg_feature_AE(x, m=3, r=0.15)  # Approximate entropy
        emg_feas.append(fea_ae)
    if is_string_in_list(feature_type, 'SE'):
        fea_se = get_emg_feature_SE(x, m=3, r=0.15)  # Sample entropy
        emg_feas.append(fea_se)
    if is_string_in_list(feature_type, 'FE'):
        fea_fe = get_emg_feature_FE(x, m=3, r=0.15, n=2)  # Fuzzy entropy
        emg_feas.append(fea_fe)

    return emg_feas
