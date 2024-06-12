import os
import math
import torch
import numpy as np
from trainTest.datasets.dataloader_shffule_utils import initDatasetShffule
from trainTest.datasets.dataloader_utils import initDataset
from utils.common_utils import calculate_class_weights_torch
from sklearn.neighbors import NearestNeighbors


def get_fileName_weights(path, gait_motion, motion, subject, subjects_list):
    if gait_motion not in ['gait', 'motion']:
        raise ValueError('gait_or_motion must be one of [gait, motion]')
    if motion not in ['WAK', 'UPS', 'DNS']:
        raise ValueError('motion_type must be one of [WAK, UPS, DNS]')
    if subject not in subjects_list:
        raise ValueError('subject not in subjects_list_global', subjects_list)
    encoded_label_name = 'sub_' + gait_motion + '_label_encoded'
    raw_label_name = 'sub_' + gait_motion + '_label_raw'
    file_name = ''
    raw_label_type = []
    if gait_motion == 'gait':
        file_name = os.path.join(path, ''.join([gait_motion, 'Classification']), motion,
                                 ''.join(['Sub', subject, '_targetTrainData.npz']))

    elif gait_motion == 'motion':
        file_name = os.path.join(path, ''.join([gait_motion, 'Classification']),
                                 ''.join(['Sub', subject, '_targetTrainData.npz']))
    with open(file_name, 'rb') as f:
        encoded_labels = np.load(f)[encoded_label_name]
        raw_labels = np.load(f)[raw_label_name]

    class_weights = calculate_class_weights_torch(encoded_labels)
    raw_label_type = list(np.unique(raw_labels))

    return file_name, class_weights, encoded_label_name, raw_label_type, encoded_labels


def get_intra_dataloaders(path, data_list, feature_list, label_name, total_exp_time, gait_or_motion,
                          current_exp_time, train_batch, test_batch, valid_batch, gait_dataset_divide_mode):
    init_dataset = None
    if gait_or_motion == 'motion' or (gait_or_motion == 'gait' and gait_dataset_divide_mode == 'random'):
        init_dataset = initDatasetShffule()
        init_dataset.initIntraSubjectDataset(path=path, raw_data_list=data_list, raw_feature_list=feature_list,
                                             label_name=label_name, total_exp_time=total_exp_time,
                                             gait_or_motion=gait_or_motion)
    elif gait_or_motion == 'gait' and gait_dataset_divide_mode in ['group_fix', 'group_random']:
        init_dataset = initDataset()
        init_dataset.initIntraSubjectDataset(path=path, raw_data_list=data_list, raw_feature_list=feature_list,
                                             label_name=label_name, total_exp_time=total_exp_time,
                                             gait_or_motion=gait_or_motion,
                                             gait_dataset_divide_mode=gait_dataset_divide_mode)

    train_loader, valid_loader, test_loader = init_dataset.getDataLoader_intra(exp_time=current_exp_time,
                                                                               train_batch=train_batch,
                                                                               test_batch=test_batch,
                                                                               valid_batch=valid_batch)
    return train_loader, valid_loader, test_loader


def get_pcc_knn_adj_from_dataloader(dataloader, params, edge_gen_mode):
    all_pcc_matrix = []
    all_knn_matrix = []
    pcc_act_thr = params['pcc_act_thr']
    kgnn_ratio, kgnn_act_thr = params['kgnn_ratio'], params['kgnn_act_thr']
    if edge_gen_mode == 'AWMF':
        for data, feature, target in dataloader:
            # [batch, 1, 15, 96]
            batch_data = data.to(device='cpu')
            # [batch, 15, 96]
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            # [batch, 15, 15]
            batch_pcc_matrix = generate_batch_pcc_matrix(batch_data)
            # [batch, 15, 15]
            batch_knn_matrix = generate_batch_knn_matrix(batch_data, kgnn_ratio)
            all_pcc_matrix.extend(batch_pcc_matrix)
            all_knn_matrix.extend(batch_knn_matrix)
        # all_pcc_matrix / all_knn_matrix: [samplesize, 15, 15]
        # pcc_adj / knn_adj : [15, 15]
        pcc_adj = np.mean(all_pcc_matrix, axis=0)
        knn_adj = np.mean(all_knn_matrix, axis=0)
        # 激活操作
        if 0 <= pcc_act_thr <= 1 and 0 <= kgnn_act_thr <= 1 :
            act_pcc_adj = np.where(pcc_adj >= pcc_act_thr, 1, 0).astype('int')
            act_knn_adj = np.where(knn_adj >= kgnn_act_thr, 1, 0).astype('int')
        else:
            raise ValueError('pcc_act_thr or kgnn_act_thr must be in [0, 1]')
        # 统计edge_numer, for learnable edge_weight
        pcc_edge_num, knn_edge_num = np.sum(act_pcc_adj == 1), np.sum(act_knn_adj == 1)
        pcc_adj, knn_adj = torch.from_numpy(act_pcc_adj), torch.from_numpy(act_knn_adj)
        pcc_knn_adjs = {'pcc_adj': pcc_adj, 'pcc_edge_num': pcc_edge_num,
                        'knn_adj': knn_adj, 'knn_edge_num': knn_edge_num}
    elif edge_gen_mode == 'PCC':
        for data, feature, target in dataloader:
            batch_data = data.to(device='cpu')
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            batch_pcc_matrix = generate_batch_pcc_matrix(batch_data)
            all_pcc_matrix.extend(batch_pcc_matrix)
        pcc_adj = np.mean(all_pcc_matrix, axis=0)
        if 0 <= pcc_act_thr <= 1:
            act_pcc_adj = np.where(pcc_adj >= pcc_act_thr, 1, 0).astype('int')
        else:
            raise ValueError('pcc_act_thr must be in [0, 1]')
        pcc_edge_num = np.sum(act_pcc_adj == 1)
        pcc_adj = torch.from_numpy(act_pcc_adj)
        pcc_knn_adjs = {'pcc_adj': pcc_adj, 'pcc_edge_num': pcc_edge_num}
    else:
        for data, feature, target in dataloader:
            batch_data = data.to(device='cpu')
            batch_data = torch.squeeze(batch_data, dim=1).numpy()
            batch_knn_matrix = generate_batch_knn_matrix(batch_data, kgnn_ratio)
            all_knn_matrix.extend(batch_knn_matrix)
        knn_adj = np.mean(all_knn_matrix, axis=0)
        if 0 <= kgnn_act_thr <= 1:
            act_knn_adj = np.where(knn_adj >= kgnn_act_thr, 1, 0).astype('int')
        else:
            raise ValueError('kgnn_act_thr must be in [0, 1]')
        knn_edge_num = np.sum(act_knn_adj == 1)
        knn_adj = torch.from_numpy(act_knn_adj)
        pcc_knn_adjs = {'knn_adj': knn_adj, 'knn_edge_num': knn_edge_num}

    return pcc_knn_adjs


def generate_batch_pcc_matrix(batch_data):
    batch_pcc_matrix = []
    for i in range(batch_data.shape[0]):
        # [1, 15, 96]
        data = batch_data[i, :, :]
        # [15, 96]
        # print(data.shape)
        # data = np.squeeze(data, axis=0)
        # [15, 15]
        pcc_matrix = np.abs(np.corrcoef(data))
        # [batch, 15, 15]
        batch_pcc_matrix.append(pcc_matrix)

    return np.array(batch_pcc_matrix)


def generate_batch_knn_matrix(batch_data, kgnn_ratio):
    node_amount = batch_data.shape[1]
    k_neighbors = math.floor(kgnn_ratio*node_amount)
    if k_neighbors < 1:
        raise ValueError('k_neighbors < 1, please increase kgnn_ratio')
    elif k_neighbors > node_amount:
        raise ValueError('k_neighbors > node_amount, please decrease kgnn_ratio')
    else:
        batch_knn_matrix = []
        for i in range(batch_data.shape[0]):
            # [1, 15, 96]
            data = batch_data[i, :, :]
            # [15, 96]
            # data = np.squeeze(data, axis=0)
            knn_graph = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
            knn_graph.fit(data)
            distances, indices = knn_graph.kneighbors(data)
            # 构建KNN图 [15, 15]
            knn_matrix = np.zeros((node_amount, node_amount), dtype=int)
            for k in range(node_amount):
                for neighbor in indices[k]:
                    knn_matrix[k][neighbor] = 1

            batch_knn_matrix.append(knn_matrix)

        return np.array(batch_knn_matrix)


def get_print_info(gait_motion, motion, subjects_list):
    info = ''
    if gait_motion == 'motion':
        info = ['当前任务：下肢运动识别，总受试者：', subjects_list]
    else:
        info = ['当前任务：%s运动下的步态相位识别，总受试者' % motion, subjects_list]

    return info


def get_save_path(base_path, gait_motion, motion, model_name, subject):
    absolute_path, relative_path = '', ''
    if gait_motion == 'gait':
        absolute_path = os.path.join(base_path, ''.join([gait_motion, 'Classification']), motion, model_name,
                                     ''.join(['Sub', subject]))
        relative_path = os.path.relpath(absolute_path, base_path)
    elif gait_motion == 'motion':
        absolute_path = os.path.join(base_path, ''.join([gait_motion, 'Classification']), model_name,
                                     ''.join(['Sub', subject]))
        relative_path = os.path.relpath(absolute_path, base_path)

    return {'absolute_path': absolute_path, 'relative_path': relative_path}
