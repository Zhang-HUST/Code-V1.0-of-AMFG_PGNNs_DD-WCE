import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
# font = {'weight': 'normal', 'size': 18}
# font1 = {'weight': 'normal', 'size': 16}


"""1. 针对CNNs模型的tsne"""


def get_cnns_features_from_dataloader(model, data_loader, feature_type, branch):
    model = model.to(device='cpu')
    deep_features_list = []
    label_list = []
    for data, feature, label in data_loader:
        deep_feature = get_deep_features_from_cnns(model, data, feature, feature_type, branch)
        # deep_feature = nn.Softmax()(deep_feature)
        deep_features_list.append(deep_feature)
        label_list.append(label)

    deep_features = torch.cat(deep_features_list, dim=0).detach().numpy()
    labels = torch.cat(label_list, dim=0).view(-1).numpy()

    return deep_features, labels


def get_deep_features_from_cnns(model, data, feature, feature_type, branch):
    deep_feature = None
    if feature_type == 'Linear':
        deep_feature = model(data, feature)
        deep_feature = nn.Softmax(dim=1)(deep_feature)
    elif feature_type == 'CNN':
        if model.get_model_name() == 'OneBranchDNN':
            deep_feature = feature.view(*(feature.size(0), -1))
        else:
            # data: [32, 1, 15, 96], feature: [32, 1, 15, 6]
            batch_size = data.shape[0]
            # [32, 96, 15, 1]
            cnn_out_1 = model.cnn_part1(data)
            cnn_out_2 = model.cnn_part2(cnn_out_1)
            cnn_out_2 = cnn_out_2.reshape(batch_size, -1)
            if branch == 1:
                deep_feature = cnn_out_2
            elif branch == 2:
                new_feature = feature.view(*(feature.size(0), -1))
                # torch.Size([32, 256+90])
                deep_feature = torch.cat([cnn_out_2, new_feature], dim=1)
    else:
        raise Exception('feature_type wrong ! support Linear and CNN only!')

    return deep_feature


class TsneVisualizationCNNs:
    def __init__(self, model, data_loader, feature_type, branch, save_results, save_fig, result_name, fig_name,
                 num_classes=None):

        self.save_fig, self.save_results = save_fig, save_results
        self.fig_name, self.result_name = fig_name, result_name
        self.feature_type = feature_type
        self.num_classes = num_classes
        self.model, self.data_loader, self.branch = model, data_loader, branch

    def init(self):
        self.model.eval()
        deep_features, labels = get_cnns_features_from_dataloader(self.model, self.data_loader, self.feature_type,
                                                                  self.branch)
        perplexity = 30 if len(labels) > 30 else len(labels)-1
        tsne = TSNE(n_components=2, perplexity=perplexity)
        if self.num_classes is not None:
            num_classes = self.num_classes
        else:
            num_classes = np.unique(labels)
        features_embedded = tsne.fit_transform(deep_features)

        if self.save_results:
            data_save = pd.DataFrame(np.concatenate([labels.reshape(-1, 1), features_embedded], axis=1))
            data_save.columns = ['labels', 'features_d0', 'features_d1']
            data_save.to_csv(self.result_name, index=True)
        self.show_tsne(features_embedded, labels, num_classes)

    def show_tsne(self, features, labels, num_classes):
        palettes = sns.color_palette("bright", num_classes)
        sns.set(rc={'figure.figsize': (12, 8)})
        plt.figure(figsize=(12, 8), dpi=300, constrained_layout=True, )
        plt.title('TSNE visualization of %s layer out' % self.feature_type, font)
        sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, style=labels, size=labels, legend='full',
                        palette=palettes)
        plt.xlabel('Feature value of dimension 1', font1)
        plt.ylabel('Feature value of dimension 2', font1)
        if self.save_fig:
            plt.savefig(self.fig_name, dpi=300, format='jpg')
        plt.show()


"""2. 针对GNNs模型的tsne"""


def get_gnns_features_from_dataloader(model, data_loader):
    model = model.to(device='cpu')
    deep_features_list = []
    label_list = []
    for data, feature, label in data_loader:
        deep_feature = model(data, feature)
        deep_feature = nn.Softmax(dim=1)(deep_feature)
        deep_features_list.append(deep_feature)
        label_list.append(label)

    deep_features = torch.cat(deep_features_list, dim=0).detach().numpy()
    labels = torch.cat(label_list, dim=0).view(-1).numpy()

    return deep_features, labels


class TsneVisualizationGNNs:
    def __init__(self, model, data_loader, save_results, save_fig, result_name, fig_name,
                 num_classes=None):

        self.save_fig, self.save_results = save_fig, save_results
        self.fig_name, self.result_name = fig_name, result_name
        self.num_classes = num_classes
        self.model, self.data_loader = model, data_loader

    def init(self):
        self.model.eval()
        deep_features, labels = get_gnns_features_from_dataloader(self.model, self.data_loader)
        perplexity = 30 if len(labels) > 30 else len(labels) - 1
        tsne = TSNE(n_components=2, perplexity=perplexity)
        if self.num_classes is not None:
            num_classes = self.num_classes
        else:
            num_classes = np.unique(labels)
        features_embedded = tsne.fit_transform(deep_features)

        if self.save_results:
            data_save = pd.DataFrame(features_embedded, labels)
            data_save.to_csv(self.result_name, index=False, header=None)
        self.show_tsne(features_embedded, labels, num_classes)

    def show_tsne(self, features, labels, num_classes):
        palettes = sns.color_palette("bright", num_classes)
        sns.set(rc={'figure.figsize': (12, 8)})
        plt.figure(figsize=(12, 8), dpi=300, constrained_layout=True, )
        plt.title('TSNE visualization of readout layer out', font)
        sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, style=labels, size=labels, legend='full',
                        palette=palettes)
        plt.xlabel('Feature value of dimension 1', font1)
        plt.ylabel('Feature value of dimension 2', font1)
        if self.save_fig:
            plt.savefig(self.fig_name, dpi=300, format='jpg')
        plt.show()


"""3. 简单通用的tsne"""


class TsneVisualization:
    def __init__(self, features, labels, num_classes=None):
        # feture: [samplesize, dim]
        self.features, self.labels = features, labels
        self.num_classes = num_classes
        self.features_embedded = None
        self.init()

    def init(self):
        perplexity = 30 if len(self.labels) > 30 else len(self.labels) - 1
        tsne = TSNE(n_components=2, perplexity=perplexity)
        # features_embedded: [samplesize, 2]
        self.features_embedded = tsne.fit_transform(self.features)

    def show_tsne(self, save_fig, fig_name: str):
        if self.num_classes is not None:
            num_classes = self.num_classes
        else:
            num_classes = np.unique(self.labels)

        if self.features_embedded is not None:
            palettes = sns.color_palette("bright", num_classes)
            sns.set(rc={'figure.figsize': (12, 8)})
            plt.figure(figsize=(12, 8), dpi=300, constrained_layout=True, )
            plt.title('TSNE visualization of readout layer out', font)
            sns.scatterplot(x=self.features_embedded[:, 0], y=self.features_embedded[:, 1],
                            hue=self.labels, style=self.labels, size=self.labels, legend='full',
                            palette=palettes)
            plt.xlabel('Feature value of dimension 1', font1)
            plt.ylabel('Feature value of dimension 2', font1)
            if save_fig:
                plt.savefig(fig_name, dpi=300, format='jpg')
            plt.show()
        else:
            raise Exception('Please use init() first!')

    def save_results(self, result_name: str):
        if self.features_embedded is not None:
            data_save = pd.DataFrame(self.features_embedded, self.labels)
            data_save.to_csv(result_name, index=False, header=None)
        else:
            raise Exception('Please use init() first!')


