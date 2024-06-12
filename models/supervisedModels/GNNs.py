import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import knn_graph
from torch_geometric.utils import dense_to_sparse
from models.commonBlocks.CNNLinearBlocks import CNNBlock, LinearBlock
from models.commonBlocks.SKConvBlock import SKBlock
from models.commonBlocks.ResNetBlocks import ResNetBlock
from models.commonBlocks.MobileNetBlocks import MobileNetBlock
from models.commonBlocks.ResNeStBlock import ResNeStBlock
from models.commonBlocks.ShffleNetBlocks import ShuffleNetBlock
from models.configs.GNNsConfigs import (common_configs, cnn_configs, sk_configs, resnet_configs, mobilenet_configs,
                                        resnest_configs, shfflenet_configs, gnn_configs, linear_configs)


class GeneralGNNs(nn.Module):
    def __init__(self, network_branch, gnn_mode, edge_gen_mode, edge_weight_mode, graph_fusion_mode, readout_mode,
                 conv_type, node_amount, gait_or_motion, motion_type, psk_path, params):
        super().__init__()
        self.network_branch = network_branch
        self.gnn_mode = gnn_mode
        # 空间域： 1）GATConv (2017); 2) TransformerConv (2020); 3) GATv2Conv (2021); 4) SuperGATConv (2021)
        # 谱域： 5）GCNConv (2016); 6) ChebConv (2016);  7) LEConv (2019); 8)SSGConv (2021).
        self.edge_gen_mode = edge_gen_mode
        # 1) 'PSK'; 2) 'PCC'; 3) 'KNN'; 4) 'TRG'; 5) 'AWMF'.
        self.edge_weight_mode = edge_weight_mode
        # 1) 'default'(对GCNs, 边权全为1；对GATs, 无);
        # 2) 'learnable' (对GCNs, TRG和PSK的边权为可学习的参数，PCC和AWMF的边权由adj的系数给出；对GATs, 无).
        self.readout_mode = readout_mode
        # 1) 'max'; 2)'mean'; 2) 'fc'.
        self.conv_type = conv_type
        self.node_amount = node_amount

        # TRG的参数设置
        if self.edge_gen_mode in ['TRG', 'AWMF']:
            self.max_esr_ratio = params['max_esr_ratio']  # TRG图中保留的边比例
            # math.floor / int： 向下取整；math.ceil： 向上取整；round： 四舍五入取整
            self.max_edge_number = math.floor(self.max_esr_ratio * self.node_amount)
            # 'graph_fix' and 'node_fix'为max_edge_number; 'graph_random'为random.choice(list(range(1, max_edge_number)))
            self.trg_edge_connect_mode = params['trg_edge_connect_mode']
            self.node_embedding = params['node_embedding']
        # 其他构图方式的参数设置
        if self.edge_gen_mode in ['PCC', 'KNN', 'AWMF']:
            self.pcc_kgnn_adjs = params['pcc_kgnn_adjs']
            self.pcc_kgnn_gen_mode = params['pcc_kgnn_gen_mode']
        if self.edge_gen_mode == 'PCC':
            self.pcc_act_thr = params['pcc_act_thr']  # 用于控制边连接是否激活的阈值
        if self.edge_gen_mode == 'KNN':
            self.kgnn_ratio = params['kgnn_ratio']  # KNN图中K近邻的比例
            self.kgnn_act_thr = params['kgnn_act_thr']  # 用于控制边连接是否激活的阈值
        if self.edge_gen_mode == 'AWMF':
            # PCC和KNN的参数已经寻优得到
            self.pcc_kgnn_gen_mode = 'train_set'
            self.pcc_act_thr = 0.5
            self.kgnn_act_thr = 0.3
            self.kgnn_ratio = 0.3
            # 仅对edge_gen_mode == 'AWMF'生效，1) 'default'; 2) 'learnable'.
            self.graph_fusion_mode = graph_fusion_mode
            self.AWMF_type = params['AWMF_type']
            self.AWMF_act_thr = params['awmf_act_thr']
            # 最少融合两个['PSK', 'TRG', 'PCC', 'KNN']
            # assert 'TRG' in self.AWMF_type and 4 >= len(self.AWMF_type) >= 2
            assert 4 >= len(self.AWMF_type) >= 2

        self.batch_size, self.device = None, None
        self.edge_index, self.edge_weight = None, None

        # 浅层的卷积特征提取块：
        bascic_configs = common_configs()
        (BN1, Activation1, Pooling1, Dropout1, BN2, Activation2, Pooling2, Dropout2, BN3, Activation3, Pooling3,
         Dropout3) = bascic_configs

        if self.conv_type == 'Conv':
            configs = cnn_configs()
            (Conv1, Conv2, Conv3) = configs
            self.cnnPart = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                CNNBlock(conv_dict=Conv2, bn_dict=BN2, activation_dict=Activation2, pooling_dict=Pooling2,
                         dropout_dict=Dropout2),
                CNNBlock(conv_dict=Conv3, bn_dict=BN3, activation_dict=Activation3, pooling_dict=Pooling3,
                         dropout_dict=Dropout3), )

        elif self.conv_type == 'SKConv':
            configs = sk_configs()
            (SKConv1, SKConv2, SKConv3) = configs
            self.cnnPart = nn.Sequential(
                SKBlock(skconv_dict=SKConv1, pooling_dict=Pooling1, dropout_dict=Dropout1),
                SKBlock(skconv_dict=SKConv2, pooling_dict=Pooling2, dropout_dict=Dropout2),
                SKBlock(skconv_dict=SKConv3, pooling_dict=Pooling3, dropout_dict=Dropout3), )

        elif self.conv_type in ['ResNetV1', 'ResNetV2']:
            configs = resnet_configs()
            (Conv1, ResNet_2, ResNet_3) = configs
            self.cnnPart = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ResNetBlock(resnet_dict=ResNet_2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                            block_type=self.conv_type),
                ResNetBlock(resnet_dict=ResNet_3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                            block_type=self.conv_type),
            )

        elif self.conv_type in ['MobileNetV1', 'MobileNetV2', 'MobileNetV3']:
            configs = mobilenet_configs()
            (Conv1, MobileNet_2, MobileNet_3) = configs
            self.cnnPart = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                MobileNetBlock(mobilenet_dict=MobileNet_2, bn_dict=BN2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                               block_type=self.conv_type),
                MobileNetBlock(mobilenet_dict=MobileNet_3, bn_dict=BN3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                               block_type=self.conv_type),
            )

        elif self.conv_type == 'ResNeSt':
            configs = resnest_configs()
            (Conv1, ResNeSt_2, ResNeSt_3) = configs
            self.cnnPart = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ResNeStBlock(resnestconv_dict=ResNeSt_2, pooling_dict=Pooling2, dropout_dict=Dropout2),
                ResNeStBlock(resnestconv_dict=ResNeSt_3, pooling_dict=Pooling3, dropout_dict=Dropout3),
            )

        elif self.conv_type in ['ShuffleNetV1', 'ShuffleNetV2']:
            configs = shfflenet_configs()
            (Conv1, ShuffleNet_2, ShuffleNet_3) = configs
            self.cnnPart = nn.Sequential(
                CNNBlock(conv_dict=Conv1, bn_dict=BN1, activation_dict=Activation1, pooling_dict=Pooling1,
                         dropout_dict=Dropout1),
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                                block_type=self.conv_type),
                ShuffleNetBlock(shufflenet_dict=ShuffleNet_3, pooling_dict=Pooling3, dropout_dict=Dropout3,
                                block_type=self.conv_type),
            )

        else:
            raise ValueError('conv_type must be one of [Conv, SKConv, ResNetV1, ResNetV2, MobileNetV1, MobileNetV2, '
                             'MobileNetV3, ResNeSt, ShuffleNetV1, ShuffleNetV2]')

        # 深层的图网络特征提取
        configs = gnn_configs(psk_path)
        (self.GCNs1, self.GCNs2, self.GATs1, self.GATs2, psk_matrix, psk_edge_num) = configs
        self.gnn1, self.gnn2, self.gnn_relu_dropout = self.get_gnn_layers()

        # 获取基础PSK邻接矩阵和边连接的个数
        if self.edge_gen_mode in ['PSK', 'AWMF']:
            self.basicPSKAdj, basicPSK_edge_num = psk_matrix, psk_edge_num
            self.PSK_edge_weight = nn.Parameter(torch.randn(1, basicPSK_edge_num))
        # 获取基础随机生成的邻接矩阵
        if self.edge_gen_mode in ['TRG', 'AWMF']:
            self.basicTRGAdj, basicTRG_edge_num = self.generateFromTRGBasic_node_level()
            self.TRG_edge_weight = nn.Parameter(torch.randn(1, basicTRG_edge_num))
            self.TRG_graph_edgeIndex = self.generate_random_edge_graph_level()
            self.TRG_graph_edge_weight = nn.Parameter(torch.randn(1, self.TRG_graph_edgeIndex.shape[1]))
            # 节点嵌入 for TRG and trg_edge_connect_mode = 'graph_fix'
            # if self.node_embedding:
            if self.edge_gen_mode == 'TRG' and self.node_embedding:
                self.embedding_matrix = nn.Parameter(torch.randn(self.GCNs1['in_channels'], self.GCNs1['in_channels']))
        if self.edge_gen_mode in ['PCC', 'AWMF']:
            self.basic_pcc_adj, basic_pcc_edge_num = None, None
            if self.pcc_kgnn_adjs is not None:
                self.basic_pcc_adj, basic_pcc_edge_num = self.pcc_kgnn_adjs['pcc_adj'], self.pcc_kgnn_adjs['pcc_edge_num']
                self.PCC_edge_weight = nn.Parameter(torch.randn(1, basic_pcc_edge_num))
        if self.edge_gen_mode in ['KNN', 'AWMF']:
            self.basic_knn_adj, basic_knn_edge_num = None, None
            if self.pcc_kgnn_adjs is not None:
                self.basic_knn_adj, basic_knn_edge_num = self.pcc_kgnn_adjs['knn_adj'], self.pcc_kgnn_adjs['knn_edge_num']
                self.KNN_edge_weight = nn.Parameter(torch.randn(1, basic_knn_edge_num))
        if self.edge_gen_mode == 'AWMF':
            self.AWMF_edge_gen_weight = nn.Parameter(torch.randn(1, len(self.AWMF_type)))

        # 分类器模块
        fc_configs = linear_configs(gait_or_motion, motion_type, self.GCNs2['out_channels'], network_branch,
                                    node_amount, readout_mode)
        (Linear1, BN1_linear, Activation1_linear, Dropout1_linear, Linear2, BN2_linear, Activation2_linear,
         Dropout2_linear) = fc_configs
        self.linear_part = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2, bn_dict=BN2_linear, activation_dict=Activation2_linear,
                        dropout_dict=Dropout2_linear),
        )
        self.init_params()

    def forward(self, data, feature):
        # data: [32, 1, 15, 96], feature: [32, 1, 15, 6]
        self.batch_size = data.shape[0]
        self.device = data.device
        # print(self.device)
        # cnn块的输出
        cnn_out = self.cnnPart(data)
        cnn_out = cnn_out.squeeze(3)
        cnn_out = torch.transpose(cnn_out, 1, 2).contiguous()  # batchSize * 15 *96
        nodes = cnn_out  # batchSize * 15 *96
        # 节点嵌入
        if self.edge_gen_mode == 'TRG' and self.node_embedding:
            nodes = torch.matmul(nodes, self.embedding_matrix)
        featureDim = nodes.shape[2]
        # 'batch_cnn_feature' / 'batch_train_data' / 'train_set'
        if self.edge_gen_mode in ['PCC', 'KNN']:
            if self.pcc_kgnn_gen_mode == 'batch_cnn_feature':
                # for PCC
                nodesClone_pcc = nodes.clone().detach().to(device='cpu')
                nodes = nodes.view(-1, featureDim)
                # for KNN
                nodesClone_knn = nodes.clone().detach().to(device='cpu')
            elif self.pcc_kgnn_gen_mode == 'batch_train_data':
                nodes = nodes.view(-1, featureDim)
                temp = torch.squeeze(data, dim=1)
                nodesClone_pcc = temp.clone().detach().to(device='cpu')
                temp = temp.view(-1, temp.shape[-1])
                nodesClone_knn = temp.clone().detach().to(device='cpu')
            elif self.pcc_kgnn_gen_mode == 'train_set':
                nodes = nodes.view(-1, featureDim)
                nodesClone_pcc, nodesClone_knn = None, None
            else:
                raise ValueError('pcc_kgnn_gen_mode error, ''support "batch_cnn_feature", '
                                 '"batch_train_data" and "train_set" only!')
        else:
            nodes = nodes.view(-1, featureDim)
            nodesClone_pcc, nodesClone_knn = None, None

        # print(nodesClone_pcc.size(), nodesClone_knn.size())
        # 获取边连接方式和边权
        self.edge_index, self.edge_weight = self.get_edge_index_weight(nodesClone_pcc, nodesClone_knn)
        edge_index = self.edge_index.to(device=self.device)
        edge_weight = self.edge_weight.to(device=self.device, dtype=torch.float)

        # 正式进入图网络前向传播
        if self.gnn_mode in ['GATConv', 'TransformerConv', 'GATv2Conv', 'SuperGATConv']:
            gnn_out = self.gnn1(nodes, edge_index)
            gnn_out = self.gnn_relu_dropout(gnn_out)
            gnn_out = self.gnn2(gnn_out, edge_index)
            gnn_out = self.gnn_relu_dropout(gnn_out)
        elif self.gnn_mode in ['GCNConv', 'ChebConv', 'LEConv', 'SSGConv']:
            gnn_out = self.gnn1(nodes, edge_index, edge_weight)
            gnn_out = self.gnn_relu_dropout(gnn_out)
            gnn_out = self.gnn2(gnn_out, edge_index, edge_weight)
            gnn_out = self.gnn_relu_dropout(gnn_out)
        else:
            raise ValueError('gnn_mode wrong ! support GATConv, TransformerConv, GATv2Conv, SuperGATConv, '
                             'GCNConv, ChebConv, LEConv, SSGConv only!')
        # batchSize * 15 * 128
        gnn_out = gnn_out.view(self.batch_size, self.node_amount, -1)

        # different read_out_mode
        if self.readout_mode == 'mean':
            gnn_out = torch.mean(gnn_out, dim=1, keepdim=False)
            if self.network_branch == 1:
                y = gnn_out
            elif self.network_branch == 2:
                new_feature = feature.view(*(feature.size(0), -1))
                y = torch.cat([gnn_out, new_feature], dim=1)
            else:
                raise ValueError('network_branch wrong! support 1 and 2 only!')
            out = self.linear_part(y)
        elif self.readout_mode == 'max':
            gnn_out = gnn_out.view(self.batch_size, self.node_amount, -1)
            gnn_out = torch.transpose(gnn_out, 1, 2).contiguous()
            gnn_out = nn.MaxPool2d(kernel_size=(1, self.node_amount))(gnn_out)  # batchSize * dim * 1
            gnn_out = gnn_out.squeeze(2)
            if self.network_branch == 1:
                y = gnn_out
            elif self.network_branch == 2:
                new_feature = feature.view(*(feature.size(0), -1))
                y = torch.cat([gnn_out, new_feature], dim=1)
            else:
                raise ValueError('network_branch wrong! support 1 and 2 only!')
            out = self.linear_part(y)
        elif self.readout_mode == 'fc':
            # out : [batchSize, nodeAmount * dim], 32, 15*128
            gnn_out = gnn_out.view(self.batch_size, -1)
            if self.network_branch == 1:
                y = gnn_out
            elif self.network_branch == 2:
                new_feature = feature.view(*(feature.size(0), -1))
                y = torch.cat([gnn_out, new_feature], dim=1) # [batchSize, 15*128+90]
            else:
                raise ValueError('network_branch wrong! support 1 and 2 only!')
            out = self.linear_part(y)
        else:
            raise ValueError('readout_mode wrong! support mean, max and fc only!')

        return out

    def get_gnn_layers(self):
        # if self.gnn_mode in ['GATConv', 'TransformerConv', 'GATv2Conv', 'SuperGATConv']:
        #     # bn1_dim = int(self.GATs1['out_channels'] * self.GATs1['heads']) if self.GATs1['concat'] else self.GATs1[
        #     #     'out_channels']
        #     bn2_dim = int(self.GATs2['out_channels'] * self.GATs2['heads']) if self.GATs2['concat'] else self.GATs2[
        #         'out_channels']
        #     # nn.BatchNorm1d(bn1_dim),
        #     gnn1_2 = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2))
        #     gnn2_2 = nn.Sequential(
        #         nn.BatchNorm1d(bn2_dim),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2))
        # else:
        #     # nn.BatchNorm1d(int(self.GCNs1['out_channels'])),
        #     gnn1_2 = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2))
        #     gnn2_2 = nn.Sequential(
        #         nn.BatchNorm1d(int(self.GCNs2['out_channels'])),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2))

        gnn_relu_dropout = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2))

        #  1）GATConv (2017)
        if self.gnn_mode == 'GATConv':
            gnn1 = gnn.GATConv(in_channels=self.GATs1['in_channels'], out_channels=self.GATs1['out_channels'],
                               heads=self.GATs1['heads'], concat=self.GATs1['concat'])
            gnn2 = gnn.GATConv(in_channels=self.GATs2['in_channels'], out_channels=self.GATs2['out_channels'],
                               heads=self.GATs2['heads'], concat=self.GATs2['concat'])
        #  2) TransformerConv (2020)
        elif self.gnn_mode == 'TransformerConv':
            gnn1 = gnn.TransformerConv(in_channels=self.GATs1['in_channels'], out_channels=self.GATs1['out_channels'],
                                       heads=self.GATs1['heads'], concat=self.GATs1['concat'])
            gnn2 = gnn.TransformerConv(in_channels=self.GATs2['in_channels'], out_channels=self.GATs2['out_channels'],
                                       heads=self.GATs2['heads'], concat=self.GATs2['concat'])
        #  3) GATv2Conv (2021)
        elif self.gnn_mode == 'GATv2Conv':
            gnn1 = gnn.GATv2Conv(in_channels=self.GATs1['in_channels'], out_channels=self.GATs1['out_channels'],
                                 heads=self.GATs1['heads'], concat=self.GATs1['concat'])
            gnn2 = gnn.GATv2Conv(in_channels=self.GATs2['in_channels'], out_channels=self.GATs2['out_channels'],
                                 heads=self.GATs2['heads'], concat=self.GATs2['concat'])
        #  4) SuperGATConv (2021)
        elif self.gnn_mode == 'SuperGATConv':
            gnn1 = gnn.SuperGATConv(in_channels=self.GATs1['in_channels'], out_channels=self.GATs1['out_channels'],
                                    heads=self.GATs1['heads'], concat=self.GATs1['concat'])
            gnn2 = gnn.SuperGATConv(in_channels=self.GATs2['in_channels'], out_channels=self.GATs2['out_channels'],
                                    heads=self.GATs2['heads'], concat=self.GATs2['concat'])
        #  5）GCNConv (2016);
        elif self.gnn_mode == 'GCNConv':
            gnn1 = gnn.GCNConv(in_channels=self.GCNs1['in_channels'], out_channels=self.GCNs1['out_channels'])
            gnn2 = gnn.GCNConv(in_channels=self.GCNs2['in_channels'], out_channels=self.GCNs2['out_channels'])
        #  6) ChebConv (2016)
        elif self.gnn_mode == 'ChebConv':
            gnn1 = gnn.ChebConv(in_channels=self.GCNs1['in_channels'], out_channels=self.GCNs1['out_channels'],
                                K=self.GCNs1['cheb_K'])
            gnn2 = gnn.ChebConv(in_channels=self.GCNs2['in_channels'], out_channels=self.GCNs2['out_channels'],
                                K=self.GCNs2['cheb_K'])
        #  7) LEConv (2019)
        elif self.gnn_mode == 'LEConv':
            gnn1 = gnn.LEConv(in_channels=self.GCNs1['in_channels'], out_channels=self.GCNs1['out_channels'])
            gnn2 = gnn.LEConv(in_channels=self.GCNs2['in_channels'], out_channels=self.GCNs2['out_channels'])
        #  8)SSGConv (2021).
        elif self.gnn_mode == 'SSGConv':
            gnn1 = gnn.SSGConv(in_channels=self.GCNs1['in_channels'], out_channels=self.GCNs1['out_channels'],
                               alpha=self.GCNs1['ssg_alpha'], K=self.GCNs1['ssg_K'])
            gnn2 = gnn.SSGConv(in_channels=self.GCNs2['in_channels'], out_channels=self.GCNs2['out_channels'],
                               alpha=self.GCNs2['ssg_alpha'], K=self.GCNs2['ssg_K'])
        else:
            raise ValueError('gnn_mode wrong ! support GATConv, TransformerConv, GATv2Conv, SuperGATConv, '
                             'GCNConv, ChebConv, LEConv, SSGConv only!')

        # return gnn1_1, gnn1_2, gnn2_1, gnn2_2
        return gnn1, gnn2, gnn_relu_dropout

    def generateFromTRGBasic_node_level(self):
        if self.max_edge_number < 1:
            raise ValueError('max_edge_number < 1, please increase max_esr_ratio')
        elif self.max_edge_number > self.node_amount:
            raise ValueError('max_edge_number > node_amount, please decrease max_esr_ratio')
        else:
            basicTRGAdj = torch.zeros(self.node_amount, self.node_amount)
            for k in range(self.node_amount):
                temp = [i for i in range(self.node_amount)]
                random.shuffle(temp)
                if self.trg_edge_connect_mode == 'node_fix':
                    temp = temp[0: self.max_edge_number]
                elif self.trg_edge_connect_mode == 'node_random':
                    if self.max_edge_number == 1:
                        temp = temp[0: self.max_edge_number]
                    else:
                        temp = temp[0: random.choice(list(range(1, self.max_edge_number)))]

                basicTRGAdj[k, temp] = 1
            TRG_edge_num = torch.sum(basicTRGAdj == 1).item()

            return basicTRGAdj, TRG_edge_num

    def generate_random_edge_graph_level(self):
        # ratio: 边的随机保留率
        ratio = self.max_esr_ratio
        startNode = [[i] * self.node_amount for i in range(self.node_amount)]
        startNode = np.array(startNode).reshape(-1)
        endNode = [i for i in range(self.node_amount)] * self.node_amount
        totalEdge = len(endNode)
        endNode = np.array(endNode).reshape(-1)
        edgeList = [k for k in range(totalEdge)]
        random.shuffle(edgeList)
        endIndex = math.floor(totalEdge * ratio)
        edgeList = edgeList[0: endIndex]
        startNode = startNode[edgeList]
        endNode = endNode[edgeList]
        TRG_graph_edgeIndex = np.concatenate([startNode.reshape(1, -1), endNode.reshape(1, -1)], axis=0)
        TRG_graph_edgeIndex = torch.from_numpy(TRG_graph_edgeIndex)
        return TRG_graph_edgeIndex

    def get_edge_index_weight(self, nodes_pcc, nodes_knn):
        if (self.gnn_mode in ['GATConv', 'TransformerConv', 'GATv2Conv', 'SuperGATConv'] and
                self.edge_weight_mode == 'learnable' and self.edge_gen_mode in ['PSK', 'TRG', 'PCC', 'KNN']):
            raise ValueError('GATs do not support edge_weight_mode of learnable '
                             'in the edge_gen_mode of ["PSK", "TRG", "PCC, “KNN”]!')
        elif (self.gnn_mode in ['GCNConv', 'ChebConv', 'LEConv', 'SSGConv'] and self.edge_weight_mode == 'learnable'
                and self.edge_gen_mode in ['KNN', 'PCC'] and self.pcc_kgnn_gen_mode in ['batch_cnn_feature',
                                                                                        'batch_train_data']):
            raise ValueError('Only GCNs + PCC/KNN + pcc_kgnn_gen_mode == "train_set" '
                             'support edge_weight_mode of learnable!')
        else:
            if self.edge_gen_mode == 'PSK':
                PSKAdj = self.generate_from_basic_adj(self.basicPSKAdj)
                # 邻接矩阵都为0和1，没必要激活
                edge_index, edge_weight = self.transAdj2EdgeIndex(PSKAdj, act_thr=None)
            elif self.edge_gen_mode == 'PCC':
                if nodes_pcc is not None:
                    # 此时为pcc_kgnn_gen_mode= 'batch_cnn_feature' / 'batch_train_data'的模式，需要激活
                    pearsonAdj = self.generateFromPearson(nodes_pcc)
                    # Note: 这里的皮尔森系数没有归一化
                    edge_index, edge_weight = self.transAdj2EdgeIndex(pearsonAdj, act_thr=self.pcc_act_thr)
                else:
                    # 此时为pcc_kgnn_gen_mode= 'train_set'的模式，已经在初始化的时候激活过，邻接矩阵都为0和1，没必要激活
                    pearsonAdj = self.generate_from_basic_adj(self.basic_pcc_adj)
                    edge_index, edge_weight = self.transAdj2EdgeIndex(pearsonAdj, act_thr=None)
            elif self.edge_gen_mode == 'KNN':
                if nodes_knn is not None:
                    edge_index = self.get_kgnn_index(nodes_knn)
                    edge_weight = torch.ones(1, edge_index.shape[1]).squeeze(0)
                else:
                    # 此时为pcc_kgnn_gen_mode= 'train_set'的模式，已经在初始化的时候激活过，邻接矩阵都为0和1，没必要激活
                    knnAdj = self.generate_from_basic_adj(self.basic_knn_adj)
                    edge_index, edge_weight = self.transAdj2EdgeIndex(knnAdj, act_thr=None)
            elif self.edge_gen_mode == 'TRG':
                if self.trg_edge_connect_mode in ['node_fix', 'node_random']:
                    TRGAdj = self.generateFromTRG_node_level()
                    # 邻接矩阵都为0和1，没必要激活
                    edge_index, edge_weight = self.transAdj2EdgeIndex(TRGAdj, act_thr=None)
                elif self.trg_edge_connect_mode == 'graph_fix':
                    edge_index, edge_weight = self.get_batch_edge_graph_level()
                else:
                    raise ValueError('trg_edge_connect_mode wrong ! support graph_fix, node_fix and node_random only!')
            elif self.edge_gen_mode == 'AWMF':
                all_adjs = []
                # [batch, 15, 15]
                if 'TRG' in self.AWMF_type:
                    TRGAdj = self.generateFromTRG_node_level()
                    all_adjs.append(TRGAdj)
                if 'PSK' in self.AWMF_type:
                    PSKAdj = self.generate_from_basic_adj(self.basicPSKAdj)
                    all_adjs.append(PSKAdj)
                if 'PCC' in self.AWMF_type:
                    PCCAdj = self.generate_from_basic_adj(self.basic_pcc_adj)
                    all_adjs.append(PCCAdj)
                if 'KNN' in self.AWMF_type:
                    KNNAdj = self.generate_from_basic_adj(self.basic_knn_adj)
                    all_adjs.append(KNNAdj)

                # 获取融合图 对GATs，不支持'learnable'的融合方式：
                # if self.gnn_mode in ['GATConv', 'TransformerConv', 'GATv2Conv',
                #                      'SuperGATConv'] and self.graph_fusion_mode == 'learnable':
                #     raise ValueError('GATs do onot support graph_fusion_mode of learnable!')
                # else:
                if self.graph_fusion_mode == 'learnable':
                    learn_AWMF_weight = nn.Softmax(dim=1)(self.AWMF_edge_gen_weight).to(device='cpu')
                    temp = torch.zeros(self.batch_size, self.node_amount, self.node_amount)
                    for i in range(len(self.AWMF_type)):
                        temp = all_adjs[i] * learn_AWMF_weight[0, i] + temp
                    fuseAdj = temp
                    # normalized_fuseAdj = torch.from_numpy(
                    #     (fuseAdj.numpy() - np.min(fuseAdj.numpy())) / (np.max(fuseAdj.numpy()) -
                    #                                                    np.min(fuseAdj.numpy())))
                    edge_index, edge_weight = self.transAdj2EdgeIndex(fuseAdj, act_thr=self.AWMF_act_thr)
                elif self.graph_fusion_mode == 'default':
                    temp = torch.zeros(self.batch_size, self.node_amount, self.node_amount)
                    for i in range(len(self.AWMF_type)):
                        temp = all_adjs[i] + temp
                    fuseAdj = temp / len(self.AWMF_type)
                    edge_index, edge_weight = self.transAdj2EdgeIndex(fuseAdj, act_thr=self.AWMF_act_thr)
                else:
                    raise ValueError('graph_fusion_mode wrong! support learnable and default only!')
            else:
                raise ValueError('edge_gen_mode wrong! support PSK, PCC, KNN, TRG, AWMF only!')

            return edge_index, edge_weight

    def get_kgnn_index(self, nodes):
        batch = []
        for k in range(self.batch_size):
            batch += [k] * self.node_amount
        batch = torch.tensor(batch)
        edgeIndex = knn_graph(nodes, k=math.floor(self.node_amount * self.kgnn_ratio), batch=batch, loop=True,
                              cosine=False)
        # k(int): 邻居数。
        # batch (torch.Tensor, optional): Batch vector
        # loop(bool，可选):如果设置为True，图形将包含自循环。(默认值:False)
        # flow(str, optional): 与消息传递结合使用时的流程方向(“source_to_target”或“target_to_source”)。(默认source_to_target)
        # cosine(bool，可选):如果为真，将使用余弦距离而不是欧氏距离来寻找最近的邻居。(默认值:False)
        # RuntimeError: `cosine` argument not supported on CPU
        # num_workers(int, optional): 用于计算的worker数量。如果batch不是None，或者输入在GPU上，则无效。(默认值:1)
        kgnnEdgeIndex = edgeIndex

        return kgnnEdgeIndex

    def generateFromPearson(self, nodes):
        pearson_matrixList = []
        # threshold = self.edge_act_thr
        for i in range(self.batch_size):
            features = nodes[i, :, :]
            pearson_matrix = torch.abs(torch.corrcoef(features))
            # 将大于等于 0.5 的元素设为 1，小于 0.5 的元素设为 0
            # binary_pearson_matrix = torch.from_numpy(np.where(pearson_matrix.numpy() >= threshold, 1.0, 0.0).astype('float'))
            pearson_matrixList.append(pearson_matrix.unsqueeze(0))
        person_matrix = torch.cat(pearson_matrixList, dim=0)
        return person_matrix

    def generate_from_basic_adj(self, adj):
        adjList = []
        for k in range(self.batch_size):
            adjList.append(adj.unsqueeze(0))
        return torch.cat(adjList, dim=0)

    def generateFromTRG_node_level(self):
        adjList = []
        for k in range(self.batch_size):
            adjList.append(self.basicTRGAdj.unsqueeze(0))
        return torch.cat(adjList, dim=0)

    def get_batch_edge_graph_level(self):
        batchEdgeIndex = []
        learn_edgeWeightList = []
        edgeIndex = self.TRG_graph_edgeIndex

        for sample in range(self.batch_size):
            batchEdgeIndex.append(edgeIndex + self.node_amount * sample)
            if self.edge_weight_mode == 'learnable':
                learn_edgeWeightList.append(nn.Sigmoid()(self.TRG_graph_edge_weight))
        TRG_graph_edge_index = torch.cat(batchEdgeIndex, dim=1)

        if self.edge_weight_mode == 'learnable':
            TRG_graph_edge_weight = torch.cat(learn_edgeWeightList, dim=1)
        else:
            TRG_graph_edge_weight = torch.ones(1, TRG_graph_edge_index.shape[1])

        return TRG_graph_edge_index, TRG_graph_edge_weight

    def transKgnnEdgeIndex2Adj(self, edge_index):
        adjList = []
        for _ in range(self.batch_size):
            tempadj = torch.zeros(self.node_amount, self.node_amount)
            for edgeOrder in range(len(edge_index[0])):
                tempadj[edge_index[0][edgeOrder] % self.node_amount][edge_index[1][edgeOrder] % self.node_amount] = 1
            adjList.append(tempadj.unsqueeze(0))
        batchAdj = torch.cat(adjList, dim=0)
        return batchAdj

    def transAdj2EdgeIndex(self, adj, act_thr):
        if act_thr is None:
            finalAdj = adj
        else:
            finalAdj = torch.where(adj >= act_thr, 1, 0)
        edgeIndexList = []
        edgeWeightList = []
        learn_edgeWeightList = []
        for k in range(self.batch_size):
            tempAdj = finalAdj[k, :, :]
            tempEdgeIndex, _ = dense_to_sparse(tempAdj)
            for edgeNumber in range(tempEdgeIndex.shape[1]):
                if self.gnn_mode in ['GCNConv', 'ChebConv', 'LEConv', 'SSGConv'] and self.edge_weight_mode == 'learnable':
                    if self.edge_gen_mode == 'AWMF' or (self.edge_gen_mode == 'PCC'
                                                        and self.pcc_kgnn_gen_mode in ['batch_cnn_feature', 'batch_raw_data']):
                        edgeWeightList.append(
                            adj[k][tempEdgeIndex[0][edgeNumber]][tempEdgeIndex[1][edgeNumber]].view(1, 1))
                else:
                    edgeWeightList.append(
                        finalAdj[k][tempEdgeIndex[0][edgeNumber]][tempEdgeIndex[1][edgeNumber]].view(1, 1))

            edgeIndexList.append(tempEdgeIndex + k * self.node_amount)

            # 1）对GCNs, TRG和PSK的边权为可学习的参数，PCC和AWMF的边权由adj的系数给出，对KNN，默认为1或0.；2） 对GATs, 无.
            if self.edge_weight_mode == 'learnable':
                if self.edge_gen_mode == 'PSK':
                    learn_edgeWeightList.append(nn.Sigmoid()(self.PSK_edge_weight))
                elif self.edge_gen_mode == 'TRG':
                    learn_edgeWeightList.append(nn.Sigmoid()(self.TRG_edge_weight))
                elif self.edge_gen_mode == 'KNN' and self.pcc_kgnn_gen_mode == 'train_set':
                    learn_edgeWeightList.append(nn.Sigmoid()(self.KNN_edge_weight))
                elif self.edge_gen_mode == 'PCC' and self.pcc_kgnn_gen_mode == 'train_set':
                    learn_edgeWeightList.append(nn.Sigmoid()(self.PCC_edge_weight))
                elif self.edge_gen_mode == 'AWMF':
                    learn_edgeWeightList = edgeWeightList
                else:
                    raise ValueError('edge_gen_mode must be PSK, PCC, TRG, KNN(train_set), or AWMF!')
            elif self.edge_weight_mode == 'default':
                pass
            else:
                raise ValueError('edge_weight_mode must be default or learnable')

        edgeIndex = torch.cat(edgeIndexList, dim=1)
        if len(learn_edgeWeightList) > 0:
            edgeWeight = torch.cat(learn_edgeWeightList, dim=1).squeeze(0)
        else:
            edgeWeight = torch.cat(edgeWeightList, dim=1).squeeze(0)

        return edgeIndex, edgeWeight

    def init_params(self):
        for m in self.modules():
            # , gnn.GCNConv, gnn.ChebConv, gnn.LEConv, gnn.SSGConv, gnn.GATConv, gnn.GATv2Conv,
            # gnn.TransformerConv, gnn.SuperGATConv
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                # mode选择“fan_in”保留了向前传递中权重方差的大小。选择“fan_out”保留反向传递的大小
                # 'fan_in'（默认值）：适用于使用ReLU激活函数的网络层。这个模式考虑了权重矩阵的输入单元数量（fan - in），并根据它来初始化权重。对ReLU激活函数友好。
                # 'fan_out'：适用于使用带有LeakyReLU或sigmoid等不是很常见的激活函数的网络层。这个模式考虑了权重矩阵的输出单元数量（fan - out），并根据它来初始化权重。
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_name(self):
        if self.network_branch == 1:
            basic_name = 'OneBranchGNN-'
        elif self.network_branch == 2:
            basic_name = 'TwoBranchGNN-'
        else:
            raise ValueError('network_branch wrong! support 1 and 2 only!')
        if self.edge_gen_mode == 'AWMF':
            #  '-', self.readout_mode
            model_name = ''.join([basic_name, self.conv_type, '-', self.edge_gen_mode, '-', self.graph_fusion_mode,
                                  '-', self.edge_weight_mode, '-', self.gnn_mode])
        else:
            model_name = ''.join([basic_name, self.conv_type, '-', self.edge_gen_mode, '-', self.edge_weight_mode,
                                  '-', self.gnn_mode])

        return model_name
