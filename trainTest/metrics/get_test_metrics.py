import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
from trainTest.metrics.sklearn_version import get_accuracy, get_precision, get_recall, get_f1, get_specificity, get_npv
from utils.common_utils import is_label_onehot, onehot2decimalism

# print(plt.style.available)
# 画图的整体样式设置
# 字体设置方式1
# from matplotlib.font_manager import FontProperties
# font_song = FontProperties(fname='SimSun.ttf') # 设置宋体
# font_hei = FontProperties(fname='SimHei.ttf') # 设置黑体
# plt.xlabel('横轴', fontproperties=font_song), plt.ylabel('纵轴', fontproperties=font_hei)
# plt.title('标题', fontproperties=font_song) # 示例绘图
# 字体设置方式2
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimSun'] # ['SimHei']
plt.style.use('Solarize_Light2')
font_song_global = {'family': 'SimSun', 'weight': 'normal', 'size': 16}
font_hei_global = {'family': 'SimHei', 'weight': 'normal', 'size': 16}
font_times_new_roman_global = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
# font_times_new_roman_global = {'weight': 'normal', 'size': 16}


class GetTestResults:
    def __init__(self, metrics, true_labels, predict_labels, decimal=5):
        self.metrics = metrics
        self.y_true = true_labels
        self.y_pre = predict_labels
        self.decimal = decimal
        support_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv']
        support_metrics_set = set(support_metrics)
        if not all(item in support_metrics_set for item in self.metrics):
            raise ValueError('metrics参数输入有误，请输入支持评估指标的list"[accuracy, precision, recall, f1, specificity, npv]"')

    def calculate(self):
        test_results = []
        if 'accuracy' in self.metrics:
            test_results.append(get_accuracy(self.y_true, self.y_pre, decimal=self.decimal))
        if 'precision' in self.metrics:
            test_results.append(get_precision(self.y_true, self.y_pre, decimal=self.decimal, average_type='macro'))
        if 'recall' in self.metrics:
            test_results.append(get_recall(self.y_true, self.y_pre, decimal=self.decimal, average_type='macro'))
        if 'f1' in self.metrics:
            test_results.append(get_f1(self.y_true, self.y_pre, decimal=self.decimal, average_type='macro'))
        if 'specificity' in self.metrics:
            test_results.append(get_specificity(self.y_true, self.y_pre, decimal=self.decimal, average_type='macro'))
        if 'npv' in self.metrics:
            test_results.append(get_npv(self.y_true, self.y_pre, decimal=self.decimal, average_type='macro'))

        return test_results

    def save(self, trial_time, test_results, file_name, file_name1):
        # 保存测试指标
        # current_exp_time等于1时，因为可能要重做实验，所以直接新建一个DataFrame保存，覆盖原文件
        # print(trial_time['current_exp_time'])
        if trial_time['current_exp_time'] == 1:
            test_metrics = pd.DataFrame()
            test_metrics.index = self.metrics
            test_metrics['1'] = test_results
            test_metrics.to_csv(file_name, index=True)
        # current_exp_time大于1小于总试验次数时，读取原有文件，在其基础上加入新的结果列
        elif 1 < trial_time['current_exp_time'] <= trial_time['total_exp_time']:
            df = pd.read_csv(file_name, header=0, index_col=0)
            df[str(trial_time['current_exp_time'])] = test_results
            if trial_time['current_exp_time'] < trial_time['total_exp_time']:
                df.to_csv(file_name, index=True)
            else:
                # current_exp_time等于总试验次数时，读取原有文件，在其基础上加入新的结果列，并计算平均值和标准差，在其基础上再加入新的平均值和标准差的列
                df_copy = df.copy()
                df['mean'] = df_copy.mean(axis=1)
                df['std'] = df_copy.std(axis=1)
                df = df.round(self.decimal)
                df.to_csv(file_name, index=True)
        # 保存预测标签和真实标签
        df1 = pd.DataFrame()
        df1['true_labels'] = self.y_true
        df1['predicted_labels'] = self.y_pre
        df1.to_csv(file_name1, index=True)


"""以下为混淆矩阵和归一化混淆矩阵的计算和绘图"""


class PlotConfusionMatrix:
    def __init__(self, y_true, y_pre, label_type=None, show_type='all', plot=True, save_fig=True, save_results=True,
                 cmap='YlGnBu'):
        self.y_true, self.y_pre = y_true, y_pre
        self.show_type = show_type
        if label_type is None:
            self.label_type = np.unique(y_true)
        else:
            self.label_type = label_type
        # label_type: 包含所有类别名的列表；show_type: {'cm', 'normalized_cm', 'all'}
        self.plot, self.save_fig, self.save_results = plot, save_fig, save_results
        self.cmap = cmap
        self.cm, self.normalized_cm = [], []

    def get_confusion_matrix(self, fig_save_name, file_save_name):
        if is_label_onehot(self.y_true, self.y_pre):
            y_true, y_pre = onehot2decimalism(self.y_true), onehot2decimalism(self.y_pre)
        else:
            pass
        if self.show_type == 'cm':
            self.cm = confusion_matrix(self.y_true, self.y_pre)
            print('Confusion Matrix: ')
            print(self.cm)
        elif self.show_type == 'normalized_cm':
            self.normalized_cm = 100 * self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            print('Normalized Confusion Matrix: ')
            print(self.normalized_cm)
        elif self.show_type == 'all':
            self.cm = confusion_matrix(self.y_true, self.y_pre)
            self.normalized_cm = 100 * self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            print('Confusion Matrix: ')
            print(self.cm)
            print('Normalized Confusion Matrix: ')
            print(self.normalized_cm)
        if self.plot:
            self.plot_confuse_matrix_results(fig_save_name)
        if self.save_results:
            self.save_confuse_matrix_results(file_save_name)

    def plot_confuse_matrix_results(self, fig_save_name):
        fig_format = fig_save_name.split(".")[-1]
        # labels = self.label_type
        labels = self.label_type[0: len(self.cm)]
        if self.show_type == 'all':
            fig = plt.figure(figsize=(16, 8), dpi=300, constrained_layout=True)
            gs = GridSpec(1, 2, figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            #  cmap='YlGnBu', 'Blues'
            ax = sns.heatmap(self.cm, cmap=self.cmap, fmt='.1f', cbar=True, annot=True, square=True, annot_kws={'size': 16},
                             xticklabels=labels, yticklabels=labels)
            ax.set_xlabel('Predicted type', font_times_new_roman_global)
            ax.set_ylabel('True type', font_times_new_roman_global)
            ax.set_title('Confuse Matrix', fontsize=18)
            ax1 = fig.add_subplot(gs[0, 1])
            ax1 = sns.heatmap(self.normalized_cm, fmt='.3f', cmap=self.cmap, cbar=True, annot=True, square=True,
                              annot_kws={'size': 16}, xticklabels=labels, yticklabels=labels)
            ax1.set_xlabel('Predicted type', font_times_new_roman_global)
            ax1.set_ylabel('True type', font_times_new_roman_global)
            ax1.set_title('Normalized Confuse Matrix (%)', fontsize=18)
            if self.save_fig:
                plt.savefig(fig_save_name, dpi=300, format=fig_format)
            plt.show()
        elif self.show_type == 'cm':
            fig, ax = plt.subplots(figsize=(9, 8), dpi=300, constrained_layout=True)
            ax = sns.heatmap(self.cm, cmap=self.cmap, fmt='.1f', cbar=True, annot=True, square=True, annot_kws={'size': 16},
                             xticklabels=labels, yticklabels=labels)
            ax.set_xlabel('Predicted type', font_times_new_roman_global)
            ax.set_ylabel('True type', font_times_new_roman_global)
            ax.set_title('Confuse Matrix', fontsize=18)
            if self.save_fig:
                plt.savefig(fig_save_name, dpi=300, format=fig_format)
            plt.show()
        elif self.show_type == 'normalized_cm':
            fig, ax = plt.subplots(figsize=(9, 8), dpi=300, constrained_layout=True)
            ax = sns.heatmap(self.normalized_cm, fmt='.3f', cmap=self.cmap, cbar=True, annot=True, square=True,
                             annot_kws={'size': 16},
                             xticklabels=labels, yticklabels=labels)
            ax.set_xlabel('Predicted type', font_times_new_roman_global)
            ax.set_ylabel('True type', font_times_new_roman_global)
            ax.set_title('Normalized Confuse Matrix (%)', fontsize=18)
            if self.save_fig:
                plt.savefig(fig_save_name, dpi=300, format=fig_format)
            plt.show()
        else:
            print('Plot error, show_type必须为cm或normalized_cm或all！')

    def save_confuse_matrix_results(self, file_save_name):
        if file_save_name.split(".")[-1] == 'xlsx':
            # labels = self.label_type
            labels = self.label_type[0: len(self.cm)]
            if self.show_type == 'all':
                confuse_matrix = pd.DataFrame(self.cm, index=labels, columns=labels)
                normalized_confuse_matrix = pd.DataFrame(self.normalized_cm, index=labels, columns=labels)
                # 创建一个 ExcelWriter 对象
                writer = pd.ExcelWriter(file_save_name, engine='xlsxwriter')
                # 将 DataFrame 写入不同的工作表
                confuse_matrix.to_excel(writer, sheet_name='cm', index=True)
                normalized_confuse_matrix.to_excel(writer, sheet_name='normalized_cm', index=True)
                # 保存并关闭写入器
                writer.close()
            elif self.show_type == 'cm':
                confuse_matrix = pd.DataFrame(self.cm, index=labels, columns=labels)
                writer = pd.ExcelWriter(file_save_name, engine='xlsxwriter')
                confuse_matrix.to_excel(writer, sheet_name=self.show_type, index=True)
                writer.close()
            elif self.show_type == 'normalized_cm':
                normalized_confuse_matrix = pd.DataFrame(self.normalized_cm, index=labels, columns=labels)
                writer = pd.ExcelWriter(file_save_name, engine='xlsxwriter')
                normalized_confuse_matrix.to_excel(writer, sheet_name=self.show_type, index=True)
                writer.close()
            else:
                print('Save error, show_type must be cm, normalized_cm and all！')
        else:
            print('Save error, file_save_name must be type of .xlsx！')
