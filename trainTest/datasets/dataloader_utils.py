import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

'''
    类initDataset的作用：
        initIntroSubjectDataset
        initInterSubjectDataset
    初始化时需传递的参数及其含义参数：
        initIntroSubjectDataset:
            path                 : 受试者npz文件路径
            raw_data_list        : 模态信号名称列表，以npz文件的字典规则为依据
            raw_feature_list     : 模态信号特征名称列表,可以为[]，表示不适用手动提取的特征
            label_name           : 用作label的npz文件字段名 ，应该是'..._encoded'
            total_exp_time       : 设计的总重复试验次数，步态相位识别中不起作用
            gait_or_motion       : 'motion':针对运动的，'gait'：针对步态的
            说明：
                initInroSubjectDataset等于给出了针对某一确定受试者提取train，valid，test DataSet的方法。当gait_or_motion参数为'motion'时，使用sklearn中的StratifiedShuffleSplit方法划分成
                total_exp_time组（train valid test）。  而当gait_or_motion参数为'gait'时，由于定义好了group_label 1-7 为训练， 8为验证，9-10为测试，因此只有一组通用的数据集（train valid test）。

        initInterSubjectDataset:


            说明：
        getDataLoader:
            exp_time             : 当前为第几次重复试验 ：1~total_exp_time
            train_batch          : 训练集batchSize
            test_batch           : 测试集batchSize
            valid_batch          : 验证集batchSize
'''


class initDataset:
    # 针对intra任务，读取单个受试者的npz文件，划分训练，测试，验证样本，供getDataLoader生成用于pytorch模型训练的dataLoader
    def initIntraSubjectDataset(self, path, raw_data_list, raw_feature_list, label_name,
                                total_exp_time, gait_or_motion, gait_dataset_divide_mode='group_random'):

        if gait_or_motion not in ['gait', 'motion']:
            raise Exception('gait_or_motion wrong! support motion or gait only!')
        if gait_dataset_divide_mode not in ['group_fix', 'group_random']:
            raise ValueError('gait_dataset_divide_mode must be in [random, group_fix, group_random]')

        # 函数返回 train valid test DataSet数据，同时也直接保存在self中
        self.gait_or_motion = gait_or_motion
        self.gait_dataset_divide_mode = gait_dataset_divide_mode

        if gait_or_motion == 'motion':
            test_ratio = 0.2
            valid_ratio = 0.1 / (1 - test_ratio)

            data = np.load(path)
            raw_data_container = []
            raw_feature_container = []
            if len(raw_data_list) == 0:
                raise Exception(f"you dont contain raw_time_domain_data,this is not right, check!")
            # 先将各模态信号在通道层拼接，可以通过raw_data_list选择信号模态
            for name in raw_data_list:
                raw_data_container.append(data[name])
            raw_data = np.concatenate(raw_data_container, axis=1)
            self.raw_data_time_step = raw_data.shape[2]  # 记录原始时域信号的样本点（时间步_time_step）
            # 再将个模态特征在通道层拼接，并且将原始信号与特征在time_step方向拼接。
            if len(raw_feature_list) != 0:
                for name in raw_feature_list:
                    raw_feature_container.append(data[name])
                raw_feature = np.concatenate(raw_feature_container, axis=1)
                self.total_data = np.concatenate([raw_data, raw_feature], axis=2)
            else:
                self.total_data = raw_data
            self.total_label = data[label_name]
            # 程序运行到这里，total_data,total_label就已经形成了
            self.total_data_shape = self.total_data.shape  # 记录一下shape
            self.total_data = self.total_data.reshape(self.total_data_shape[0], -1)
            self.total_label_shape = self.total_label.shape  # 记录一下shape

            # 使用sklearn将数据集按照比例7：1：2划分成train：valid：test。
            self.X_train = []
            self.X_valid = []
            self.X_test = []
            self.y_train = []
            self.y_valid = []
            self.y_test = []
            sss = StratifiedShuffleSplit(n_splits=total_exp_time, test_size=test_ratio, random_state=42)
            sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=42)
            for train_index, test_index in sss.split(self.total_data, self.total_label):
                X_trainAndValid, X_test = self.total_data[train_index], self.total_data[test_index]
                y_trainAndValid, y_test = self.total_label[train_index], self.total_label[test_index]
                self.X_test.append(X_test.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
                self.y_test.append(y_test)
                # 再将X_trainAndValid和Y_trainAndValid种划分出train和valid
                for train_index_, valid_index in sssForValid.split(X_trainAndValid, y_trainAndValid):
                    X_train, X_valid = X_trainAndValid[train_index_], X_trainAndValid[valid_index]
                    y_train, y_valid = y_trainAndValid[train_index_], y_trainAndValid[valid_index]
                self.X_train.append(X_train.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
                self.X_valid.append(X_valid.reshape(-1, self.total_data_shape[1], self.total_data_shape[2]))
                self.y_train.append(y_train)
                self.y_valid.append(y_valid)

        elif gait_or_motion == 'gait' and gait_dataset_divide_mode in ['group_fix', 'group_random']:  # 表示当前任务为步态相位识别任务
            data = np.load(path)
            raw_data_container = []
            raw_feature_container = []
            if len(raw_data_list) == 0:
                raise Exception(f"you dont contain raw_time_domain_data,this is not right, check!")
            # 先将各模态信号在通道层拼接，可以通过raw_data_list选择信号模态
            for name in raw_data_list:
                raw_data_container.append(data[name])
            raw_data = np.concatenate(raw_data_container, axis=1)
            self.raw_data_time_step = raw_data.shape[2]  # 记录原始时域信号的样本点（时间步_time_step）
            # 再将个模态特征在通道层拼接，并且将原始信号与特征在time_step方向拼接。
            if len(raw_feature_list) != 0:
                for name in raw_feature_list:
                    raw_feature_container.append(data[name])
                raw_feature = np.concatenate(raw_feature_container, axis=1)
                self.total_data = np.concatenate([raw_data, raw_feature], axis=2)
            else:
                self.total_data = raw_data
            self.total_label = data[label_name]
            # 程序运行到这里，total_data,total_label就已经形成了
            self.total_data_shape = self.total_data.shape  # 记录一下shape batchSize * channel * 96 (96+6)
            self.total_label_shape = self.total_label.shape  # 记录一下shape

            # 按照sub_group_label_raw切分数据集
            sub_group_label_raw = data['sub_group_label_raw']
            if gait_dataset_divide_mode == 'group_fix':
                print('Group number of train set, valid set and test set are: ', range(7), [8], [9, 10])
                train_data_end_index = np.argwhere(sub_group_label_raw == 7)[-1][0]
                valid_data_end_index = np.argwhere(sub_group_label_raw == 8)[-1][0]
                test_data_end_index = np.argwhere(sub_group_label_raw == 10)[-1][0]
                self.X_train = self.total_data[0:train_data_end_index, :, :]
                self.y_train = self.total_label[0:train_data_end_index]

                self.X_valid = self.total_data[train_data_end_index:valid_data_end_index, :, :]
                self.y_valid = self.total_label[train_data_end_index:valid_data_end_index]

                self.X_test = self.total_data[valid_data_end_index:test_data_end_index, :, :]
                self.y_test = self.total_label[valid_data_end_index:test_data_end_index]
            elif gait_dataset_divide_mode == 'group_random':
                train_list, valid_list, test_list = get_gait_train_test_lists(group=10)
                print('Group number of train set, valid set and test set are: ', train_list, valid_list, test_list)
                train_data_index = []
                for i in range(len(train_list)):
                    indices = np.where(sub_group_label_raw == train_list[i])[0].tolist()
                    train_data_index.extend(indices)
                valid_data_index = []
                for i in range(len(valid_list)):
                    indices = np.where(sub_group_label_raw == valid_list[i])[0].tolist()
                    valid_data_index.extend(indices)
                test_data_index = []
                for i in range(len(test_list)):
                    indices = np.where(sub_group_label_raw == test_list[i])[0].tolist()
                    test_data_index.extend(indices)

                self.X_train = self.total_data[train_data_index, :, :]
                self.y_train = self.total_label[train_data_index]

                self.X_valid = self.total_data[valid_data_index, :, :]
                self.y_valid = self.total_label[valid_data_index]

                self.X_test = self.total_data[test_data_index, :, :]
                self.y_test = self.total_label[test_data_index]

        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test

    # 针对inter任务，形成源域和目标域的train valid test DataSet数据，保存到self中

    def getDataSetFromDomain(self, domain_path_list, raw_data_list, raw_feature_list, label_name, total_exp_time,
                             gait_or_motion):
        # 该函数的作用是根据pathList 返回一个domain中的train valid test
        '''
            :param domain_path_list:  域路径列表
            :param raw_data_list:
            :param raw_feature_list:
            :param label_name:
            :param total_exp_time:
            :param gait_or_motion:
            :return:
        '''
        if gait_or_motion == 'motion':
            X_train_result = [[] for i in range(total_exp_time)]
            y_train_result = [[] for i in range(total_exp_time)]
            X_valid_result = [[] for i in range(total_exp_time)]
            y_valid_result = [[] for i in range(total_exp_time)]
            X_test_result = [[] for i in range(total_exp_time)]
            y_test_result = [[] for i in range(total_exp_time)]
            for path in domain_path_list:
                # 得到源域中每一个受试者的train valid test （每个人都有total_exp_time组试验）
                X_train, y_train, X_valid, y_valid, X_test, y_test = self.initIntraSubjectDataset(path, raw_data_list,
                                                                                                  raw_feature_list,
                                                                                                  label_name,
                                                                                                  total_exp_time,
                                                                                                  gait_or_motion)
                for i in range(total_exp_time):
                    X_train_result[i].append(X_train[i])
                    y_train_result[i].append(y_train[i])
                    X_valid_result[i].append(X_valid[i])
                    y_valid_result[i].append(y_valid[i])
                    X_test_result[i].append(X_test[i])
                    y_test_result[i].append(y_test[i])
            # 下面形成源域train valid test，一共total_exp_time组实验
            for i in range(total_exp_time):
                X_train_result[i] = np.concatenate(X_train_result[i], axis=0)
                y_train_result[i] = np.concatenate(y_train_result[i])
                X_valid_result[i] = np.concatenate(X_valid_result[i], axis=0)
                y_valid_result[i] = np.concatenate(y_valid_result[i])
                X_test_result[i] = np.concatenate(X_test_result[i], axis=0)
                y_test_result[i] = np.concatenate(y_test_result[i])
            # self.X_train,self.y_train,self.X_valid,self.y_valid,self.X_test,self.y_test = X_train_result,y_train_result,X_valid_result,y_valid_result,X_test_result,y_test_result
            return X_train_result, y_train_result, X_valid_result, y_valid_result, X_test_result, y_test_result
        elif gait_or_motion == 'gait':
            X_train_result = []
            y_train_result = []
            X_valid_result = []
            y_valid_result = []
            X_test_result = []
            y_test_result = []
            for path in domain_path_list:
                # 得到源域中每一个受试者的train valid test （每个人都有total_exp_time组试验）
                X_train, y_train, X_valid, y_valid, X_test, y_test = self.initIntraSubjectDataset(path, raw_data_list,
                                                                                                  raw_feature_list,
                                                                                                  label_name,
                                                                                                  total_exp_time,
                                                                                                  gait_or_motion)
                X_train_result.append(X_train)
                y_train_result.append(y_train.reshape(-1, 1))
                X_valid_result.append(X_valid)
                y_valid_result.append(y_valid.reshape(-1, 1))
                X_test_result.append(X_test)
                y_test_result.append(y_test.reshape(-1, 1))
            # 下面形成源域train valid test，一共total_exp_time组实验
            X_train_result = np.concatenate(X_train_result, axis=0)
            y_train_result = np.concatenate(y_train_result, axis=0)
            X_valid_result = np.concatenate(X_valid_result, axis=0)
            y_valid_result = np.concatenate(y_valid_result, axis=0)
            X_test_result = np.concatenate(X_test_result, axis=0)
            y_test_result = np.concatenate(y_test_result, axis=0)
            # self.X_train,self.y_train,self.X_valid,self.y_valid,self.X_test,self.y_test = X_train_result,y_train_result,X_valid_result,y_valid_result,X_test_result,y_test_result
            return X_train_result, y_train_result, X_valid_result, y_valid_result, X_test_result, y_test_result
        else:
            raise Exception('gait_or_motion wrong! support motion or gait only!')

    '''
        initInterSubjectDataset
    '''

    def initInterSubjectDataset(self, source_path_list, target_path_list, raw_data_list, raw_feature_list, label_name,
                                total_exp_time, gait_or_motion):
        self.gait_or_motion = gait_or_motion
        # 按照源域，目标域路径列表分别获取源域目标域的train，valid，test DataSet
        source_X_train, source_y_train, source_X_valid, source_y_valid, source_X_test, source_y_test = self.getDataSetFromDomain(
            source_path_list, raw_data_list, raw_feature_list, label_name, total_exp_time, gait_or_motion)
        target_X_train, target_y_train, target_X_valid, target_y_valid, target_X_test, target_y_test = self.getDataSetFromDomain(
            target_path_list, raw_data_list, raw_feature_list, label_name, total_exp_time, gait_or_motion)

        self.source_X_train, self.source_y_train, self.source_X_valid, self.source_y_valid, self.source_X_test, self.source_y_test = source_X_train, source_y_train, source_X_valid, source_y_valid, source_X_test, source_y_test
        self.target_X_train, self.target_y_train, self.target_X_valid, self.target_y_valid, self.target_X_test, self.target_y_test = target_X_train, target_y_train, target_X_valid, target_y_valid, target_X_test, target_y_test

    '''
        getDataLoader:
            exp_time : 当前重复实验次数
            train_batch : 训练集batchSize
            test_batch  : 测试集batchSize
            valid_batch : 验证集batchSize
    '''

    # 作用：返回三种DataSet：trainSet validSet testSet
    def getDataLoader_intra(self, exp_time, train_batch, valid_batch, test_batch):
        if self.gait_or_motion == 'motion':
            train_set = myDataset(data=self.X_train[exp_time - 1], label=self.y_train[exp_time - 1],
                                  time_step=self.raw_data_time_step)
            valid_set = myDataset(data=self.X_valid[exp_time - 1], label=self.y_valid[exp_time - 1],
                                  time_step=self.raw_data_time_step)
            test_set = myDataset(data=self.X_test[exp_time - 1], label=self.y_test[exp_time - 1],
                                 time_step=self.raw_data_time_step)
            train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_set, batch_size=valid_batch, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False)
        elif self.gait_or_motion == 'gait':
            train_set = myDataset(data=self.X_train, label=self.y_train, time_step=self.raw_data_time_step)
            valid_set = myDataset(data=self.X_valid, label=self.y_valid, time_step=self.raw_data_time_step)
            test_set = myDataset(data=self.X_test, label=self.y_test, time_step=self.raw_data_time_step)
            train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_set, batch_size=valid_batch, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False)
        return train_loader, valid_loader, test_loader

    # 为inter-subject任务设计的获取数据加载器方法
    def getDataLoader_inter(self, exp_time, source_train_batch, source_valid_batch, source_test_batch,
                            target_train_batch, target_valid_batch, target_test_batch):
        if self.gait_or_motion == 'motion':
            # 将源域DataSet封装成DataLoader
            source_train_set = myDataset(data=self.source_X_train[exp_time - 1],
                                         label=self.source_y_train[exp_time - 1], time_step=self.raw_data_time_step)
            source_valid_set = myDataset(data=self.source_X_valid[exp_time - 1],
                                         label=self.source_y_valid[exp_time - 1], time_step=self.raw_data_time_step)
            source_test_set = myDataset(data=self.source_X_test[exp_time - 1], label=self.source_y_test[exp_time - 1],
                                        time_step=self.raw_data_time_step)
            source_train_loader = DataLoader(source_train_set, batch_size=source_train_batch, shuffle=True,
                                             drop_last=True)
            source_valid_loader = DataLoader(source_valid_set, batch_size=source_valid_batch, shuffle=False)
            source_test_loader = DataLoader(source_test_set, batch_size=source_test_batch, shuffle=False)
            # 将目标域DataSet封装成DataLoader
            target_train_set = myDataset(data=self.target_X_train[exp_time - 1],
                                         label=self.target_y_train[exp_time - 1], time_step=self.raw_data_time_step)
            target_valid_set = myDataset(data=self.target_X_valid[exp_time - 1],
                                         label=self.target_y_valid[exp_time - 1], time_step=self.raw_data_time_step)
            target_test_set = myDataset(data=self.target_X_test[exp_time - 1], label=self.target_y_test[exp_time - 1],
                                        time_step=self.raw_data_time_step)
            target_train_loader = DataLoader(target_train_set, batch_size=source_train_batch, shuffle=True,
                                             drop_last=True)
            target_valid_loader = DataLoader(target_valid_set, batch_size=source_valid_batch, shuffle=False)
            target_test_loader = DataLoader(target_test_set, batch_size=source_test_batch, shuffle=False)
        elif self.gait_or_motion == 'gait':
            # 将源域DataSet封装成DataLoader
            source_train_set = myDataset(data=self.source_X_train, label=self.source_y_train,
                                         time_step=self.raw_data_time_step)
            source_valid_set = myDataset(data=self.source_X_valid, label=self.source_y_valid,
                                         time_step=self.raw_data_time_step)
            source_test_set = myDataset(data=self.source_X_test, label=self.source_y_test,
                                        time_step=self.raw_data_time_step)
            source_train_loader = DataLoader(source_train_set, batch_size=source_train_batch, shuffle=True,
                                             drop_last=True)
            source_valid_loader = DataLoader(source_valid_set, batch_size=source_valid_batch, shuffle=False)
            source_test_loader = DataLoader(source_test_set, batch_size=source_test_batch, shuffle=False)
            # 将目标域DataSet封装成DataLoader
            target_train_set = myDataset(data=self.target_X_train, label=self.target_y_train,
                                         time_step=self.raw_data_time_step)
            target_valid_set = myDataset(data=self.target_X_valid, label=self.target_y_valid,
                                         time_step=self.raw_data_time_step)
            target_test_set = myDataset(data=self.target_X_test, label=self.target_y_test,
                                        time_step=self.raw_data_time_step)
            target_train_loader = DataLoader(target_train_set, batch_size=target_train_batch, shuffle=True,
                                             drop_last=True)
            target_valid_loader = DataLoader(target_valid_set, batch_size=target_valid_batch, shuffle=False)
            target_test_loader = DataLoader(target_test_set, batch_size=target_test_batch, shuffle=False)
        return source_train_loader, source_valid_loader, source_test_loader, target_train_loader, target_valid_loader, target_test_loader


class myDataset(Dataset):
    def __init__(self, data, label, time_step):
        self.data = data
        self.label = label
        self.time_step = time_step
        if self.time_step == self.data.shape[2]:
            self.featureExist = False
        else:
            self.featureExist = True

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.featureExist:
            singleData = torch.from_numpy(self.data[item, :, 0:self.time_step]).unsqueeze(0)
            singleFeature = torch.from_numpy(self.data[item, :, self.time_step:]).unsqueeze(0)
            singleLabel = torch.from_numpy(self.label[item:item + 1]).to(dtype=torch.long).view(-1)
            return singleData, singleFeature, singleLabel
        else:
            singleData = torch.from_numpy(self.data[item, :, :]).unsqueeze(0)
            singleLabel = torch.from_numpy(self.label[item:item + 1]).to(dtype=torch.long).view(-1)
            return singleData, torch.randn(1, 1), singleLabel


'''
    RepeatDataLoaderIterator:
        

'''


class RepeatDataLoaderIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.dataset = data_loader.dataset

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)

        return batch


if __name__ == '__main__':
    # 下面是一个应用例子
    # intro-subject-gait-recognition
    personOrder = '01'
    path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + personOrder + "_targetTrainData.npz"
    raw_data_list = ['sub_emg_sample', 'sub_angle_sample']
    raw_feature_list = ['sub_emg_features', 'sub_angle_features']
    label_name = 'sub_gait_label_encoded'
    gait_or_motion = 'gait'

    total_exp_time = 6
    train_batch = 32
    valid_batch = 32
    test_batch = 32
    exp_time = 6
    init_dataset = initDataset()
    init_dataset.initIntraSubjectDataset(path=path, raw_data_list=raw_data_list, raw_feature_list=[],
                                         label_name=label_name, total_exp_time=6, gait_or_motion=gait_or_motion)
    train_loader, valid_loader, test_loader = init_dataset.getDataLoader_intra(exp_time=exp_time,
                                                                               train_batch=train_batch,
                                                                               test_batch=test_batch,
                                                                               valid_batch=valid_batch)

    # inter-subject-gait-regocnition
    source_personOrder = ['01', '02', '03']
    target_personOrder = ['04']
    source_path_list = []
    target_path_list = []
    for order in source_personOrder:
        path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + order + "_targetTrainData.npz"
        source_path_list.append(path)
    for order in target_personOrder:
        path = "../preProcessing/trainData/gaitClassification/DNS/Sub" + order + "_targetTrainData.npz"
        target_path_list.append(path)
    raw_data_list = ['sub_emg_sample', 'sub_angle_sample']
    raw_feature_list = ['sub_emg_features', 'sub_angle_features']
    label_name = 'sub_gait_label_encoded'
    gait_or_motion = 'gait'
    total_exp_time = 6
    exp_time = 6
    init_dataset = initDataset()
    init_dataset.initInterSubjectDataset(source_path_list=source_path_list, target_path_list=target_path_list,
                                         raw_data_list=raw_data_list, raw_feature_list=[], label_name=label_name,
                                         total_exp_time=6, gait_or_motion=gait_or_motion)
    source_train_loader, source_valid_loader, source_test_loader, target_train_loader, target_valid_loader, target_test_loader = init_dataset.getDataLoader_inter(
        exp_time=exp_time, source_train_batch=32, source_valid_batch=32, source_test_batch=32, target_train_batch=32,
        target_valid_batch=32, target_test_batch=32)

    # 调用RepeatDataLoaderIterator的例子：
    target_train_loader = RepeatDataLoaderIterator(target_train_loader)
    for data1, data2 in zip(source_train_loader, target_train_loader):
        print(data2[0].shape)


def get_gait_train_test_lists(group):
    # 生成1-10的list0
    list0 = list(range(1, group+1))

    # 随机挑选list0中的一个元素作为list3
    list3 = random.choice(list0)

    # 从list0中剔除list1的元素
    list1 = [list3]
    list0.remove(list3)

    # 从新的list0中随机挑选两个元素作为list2
    list2 = random.sample(list0, 2)

    # 从list0中剔除list2的元素
    for item in list2:
        list0.remove(item)

    # 对list0、list1和list2进行升序排列
    list0.sort()
    list1.sort()
    list2.sort()

    return list0, list1, list2
