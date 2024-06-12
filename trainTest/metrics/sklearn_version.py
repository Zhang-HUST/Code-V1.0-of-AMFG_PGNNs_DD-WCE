import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import explained_variance_score, mean_absolute_error  # 解释方差分数EVS，平均绝对误差MAE
from sklearn.metrics import mean_squared_error  # 当'squared=False'时，为均方根误差RMSE；当'squared=True（默认）'时，为方根误差MSE
from sklearn.metrics import r2_score  # 决定系数R2
from utils.common_utils import is_label_onehot, onehot2decimalism

"""分类任务评价指标：accuracy, precision, recall, f1, specificity, npv, confusion_matrix and normalized confusion_matrix"""


def get_specificity_npv(y1, y2):
    MCM = multilabel_confusion_matrix(y1, y2)
    specificity = []
    npv = []
    for i in range(MCM.shape[0]):
        confusion = MCM[i]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        specificity.append(TN / float(TN + FP))  # Sensitivity
        npv.append(TN / float(FN + TN))  # Negative predictive value（NPV)
    test_specificity = np.average(specificity)
    test_npv = np.average(npv)

    return test_specificity, test_npv


"""以下为6个分类任务评价指标的计算"""


### 计算准确率accuracy
def get_accuracy(y_true, y_pre, decimal=3):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    accuracy = accuracy_score(y_true, y_pre)

    return round(accuracy * 100.0, decimal)


### 计算精确率precision，又名阳性预测值positive predictive value (PPV)
def get_precision(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    precision = precision_score(y_true, y_pre, average=average_type, zero_division=0)  # "average": "micro", "macro", "samples", "weighted", "binary"

    return round(precision * 100.0, decimal)


### 计算召回率recall，又名灵敏度sensitivity
def get_recall(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    recall = recall_score(y_true, y_pre,
                          average=average_type)  # "average": "micro", "macro", "samples", "weighted", "binary"

    return round(recall * 100.0, decimal)


### 计算f1分数
def get_f1(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    f1 = f1_score(y_true, y_pre, average=average_type)  # "average": "micro", "macro", "samples", "weighted", "binary"

    return round(f1 * 100.0, decimal)


### 计算特异性specificity
def get_specificity(y_true, y_pre, decimal=3, average_type='macro'):
    specificity = None
    if average_type == 'macro':
        if is_label_onehot(y_true, y_pre):
            y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
        else:
            pass
        specificity = get_specificity_npv(y_true, y_pre)[0]
    else:
        print('Error, 仅支持 average_type为macro')

    return round(specificity * 100.0, decimal)


### 计算阴性预测值negative predictive value (NPV)
def get_npv(y_true, y_pre, decimal=3, average_type='macro'):
    npv = None
    if average_type == 'macro':
        if is_label_onehot(y_true, y_pre):
            y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
        else:
            pass
        npv = get_specificity_npv(y_true, y_pre)[1]
    else:
        print('Error, 仅支持 average_type为macro')

    return round(npv * 100.0, decimal)


"""以下为5个预测指标的计算"""


def get_evs(y_true, y_pre, decimal=4):
    evs = explained_variance_score(y_true, y_pre)
    return round(evs, decimal)


def get_mae(y_true, y_pre, decimal=4):
    mae = mean_absolute_error(y_true, y_pre)
    return round(mae, decimal)


def get_mse(y_true, y_pre, decimal=4):
    y_true, y_pre = y_true.numpy(), y_pre.numpy()
    mse = mean_squared_error(y_true, y_pre)
    return round(mse, decimal)


def get_rmse(y_true, y_pre, decimal=4):
    y_true, y_pre = y_true.numpy(), y_pre.numpy()
    rmse = mean_squared_error(y_true, y_pre, squared=False)
    return round(rmse, decimal)


def get_r2_score(y_true, y_pre, decimal=4):
    y_true, y_pre = y_true.numpy(), y_pre.numpy()
    r2_value = r2_score(y_true, y_pre, squared=False)
    return round(r2_value, decimal)
