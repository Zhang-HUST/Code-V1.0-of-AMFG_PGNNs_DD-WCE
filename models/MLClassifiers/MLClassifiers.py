import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import OneHotEncoder


class GeneralMLClassifiers:
    def __init__(self, classifier_type):
        self.classifier_type = classifier_type
        self.model = None
        # self.best_params = None
        # self.set_best_params = False
        self.model_train = False
        self.x_train, self.y_train = None, None

    def init(self):
        if self.classifier_type == 'KNN':
            self.model = KNeighborsClassifier()
        elif self.classifier_type == 'LDA':
            self.model = LinearDiscriminantAnalysis()
        elif self.classifier_type == 'SVM':
            self.model = SVC()
        elif self.classifier_type == 'RF':
            self.model = RandomForestClassifier()
        else:
            raise Exception('Unknown classifier type')

    def parameter_optimization(self, train_features, train_labels):
        if self.model is None:
            raise Exception('please run init() first!')
        else:
            parameters = self.get_parameters()
            gs = GridSearchCV(self.model, parameters, scoring='accuracy', refit=True, cv=5, verbose=1, n_jobs=-1)
            # scoring = None ：模型评价标准'f1'...
            # refit = True ：在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
            gs.fit(train_features, train_labels)
            # best_score = gs.best_score_
            # self.best_params = gs.best_params_
            best_params = gs.best_params_
            return best_params

    def set_params(self, params):
        if self.classifier_type == 'KNN':
            self.model = KNeighborsClassifier(**params)
        elif self.classifier_type == 'LDA':
            self.model = LinearDiscriminantAnalysis(**params)
        elif self.classifier_type == 'SVM':
            self.model = SVC(**params)
        elif self.classifier_type == 'RF':
            self.model = RandomForestClassifier(**params)
        else:
            raise Exception('Unknown classifier type')
        # self.model = self.model(**self.best_params)
        # self.set_best_params = True

    def get_parameters(self):
        if self.model is None:
            raise Exception('please run init() first!')
        else:
            if self.classifier_type == 'KNN':
                parameters = {'n_neighbors': range(3, 11),
                              'weights': ['uniform', 'distance'],
                              # 'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                              'algorithm': ['auto'],
                              # 'leaf_size': np.arange(10, 31, 10, dtype=int),
                              'p': [1, 2]
                              }
                # n_neighbors:默认情况下用于kneighbors查询的邻居数量。
                # weights: uniform:均匀的权重; distance :以点的距离的倒数表示权重。
                # algorithm: 用于计算最近邻居的算法:` ball_tree `将使用BallTree; ` kd_tree `将使用KDTree; ` brute `将使用暴力搜索; ` auto ` 将尝试根据传递给fit方法的值来决定最合适的算法。
                # leaf_size: 传递给BallTree或KDTree的叶的长度。这可能会影响构造和查询的速度，以及存储树所需的内存。最优值取决于问题的性质。
                # p - Minkowski度量的功率参数。当p = 1时，这相当于在p = 2时使用manhattan距离(l1)和euclidean_distance (l2)。对于任意的p，使用minkowski_distance (l_p)。
                # metric:用于距离计算的度量单位。默认值为"minkowski"，即p = 2时的标准欧氏距离。

            elif self.classifier_type == 'LDA':
                parameters = {'solver': ['svd', 'lsqr', 'eigen'], 'n_components': [1, 2]}
                # 'solver', 要使用的求解器: ` svd `: 奇异值分解(默认值)。` lsqr `: 最小二乘解。 ` eigen `: 特征值分解。
                # 'shrinkage' 取值为: None: 无收缩(默认)。 'auto': 使用Ledoit - Wolf引理自动收缩。浮动在0和1之间: 固定收缩参数。
                # 如果使用covariance_estimator，则该值应为None。注意，shrinkage仅适用于 ` lsqr ` 和` eigen ` 求解器。
                # priors: 类别先验概率。默认情况下，类别比例是从训练数据中推断出来的。
                # n_components - 分量的数量( & lt;= min(n_classes - 1, n_features))用于降维。如果为None，则设置为min(n_classes - 1, n_features)。这个参数只影响transform方法。
                # store_covariance - 如果为真，当求解器是` svd `时，显式计算加权的类内协方差矩阵。对于其他求解器，矩阵总是被计算并存储。
                #  tol—认为X的奇异值是显著的绝对阈值，用于估计X的秩。奇异值不显著的维度被丢弃。仅当求解器为` svd `时使用。
                # covariance_estimator: 如果不是None，则使用covariance_estimator来估计协方差矩阵，而不是依赖经验协方差估计器(可能会有收缩)。
                # 该对象应该具有fit方法和covariance_属性，类似于sklearn.covariance中的估计器。如果没有，则收缩参数驱动估计。
                # 如果使用收缩，则应将此保留为零。注意，covariance_estimator仅适用于` lsqr ` 和` eigen `求解器。

            elif self.classifier_type == 'SVM':
                parameters = {'gamma': ['scale', 0.001, 0.01, 0.1, 1, 10, 100],
                              'C': [1, 10, 50, 100, 500],
                              'max_iter': [-1]}
                # C - 正则化参数。正则化的强度与c成反比，必须严格为正。惩罚是l2惩罚的平方。
                # kernel - default = 'rbf'指定算法中使用的核函数类型。
                # degree: 多项式核函数(` poly `)的度数。必须是非负的。被所有其他核函数忽略。
                # 如果传入gamma = 'scale'(默认值)，则使用1 / (n_features * X.var())作为gamma的值，
                # max_iter: 求解器中的迭代次数的硬限制，-1示没有限制。
                # decision_function_shape - 是返回一个形状为(n_samples, n_classes)的一对其余(` ovr `)决策函数作为所有其他分类器，还是返回原始的一对一(` ovo `)
                # 总是用作多类策略来训练模型;ovr矩阵仅由ovo矩阵构造而成。对于二分类，该参数将被忽略。
            elif self.classifier_type == 'RF':
                parameters = {'n_estimators': [100, 120, 140, 160, 180, 200],
                              'max_depth': range(7, 14)}
            else:
                raise Exception('Unknown classifier type')

            return parameters

    def get_model_name(self):
        model_name = self.classifier_type
        return model_name

    def train(self, x_train, y_train):
        if self.model is not None:
            self.x_train, self.y_train = x_train, y_train
            print(x_train.shape, y_train.shape)
            self.model.fit(x_train, y_train)
            self.model_train = True
        else:
            raise Exception('未进行模型初始化！')

    def predict(self, x_test):
        if self.model_train:
            pre_y_train = self.model.predict(self.x_train)
            pre_y_test = self.model.predict(x_test)
            # if self.classifier_type in ['LDA', 'RF']:
            #     # 形状ndarray （n_samples， n_classes）
            #     pre_y_train_proba = self.model.predict_log_proba(self.x_train)
            #     pre_y_test_proba = self.model.predict_log_proba(x_test)
            #     return pre_y_train, pre_y_test, pre_y_train_proba, pre_y_test_proba
            # else:
            #     return pre_y_train, pre_y_test
            # [[-708.39641853 - 708.39641853    0. - 118.02903397 - 708.39641853]
            #  [-708.39641853 - 708.39641853 - 132.31166046    0. - 708.39641853]
            # [0. - 417.75903222 - 708.39641853 - 708.39641853 - 180.68789945]
            # ...
            # [0. - 354.73156859 - 708.39641853 - 708.39641853 - 138.42797675]
            # [-708.39641853 - 708.39641853
            # 0. - 102.23904243 - 708.39641853]
            # [-708.39641853 - 708.39641853 - 133.41107933    0. - 708.39641853]]
            return pre_y_train, pre_y_test
        else:
            raise Exception('未进行模型训练！')

    def save(self, save_name):
        if self.model_train:
            with open(save_name, 'wb') as f:
                pickle.dump(self.model, f)
            # 读取模型
            # with open('train_model.pkl', 'rb') as f:
            #     gbr = pickle.load(f)
        else:
            raise Exception('未进行模型训练！')











