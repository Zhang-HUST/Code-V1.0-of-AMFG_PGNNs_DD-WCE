# 导入小工具
from utils.common_utils import make_dir
# 导入模型分类性能评价工具
from trainTest.metrics.get_test_metrics import GetTestResults, PlotConfusionMatrix


def train_test_intra_ml_model(settings_dict, model, x_train, y_train, x_test, y_test, save_path, utils):
    # 1. 模型、测试结果保存绝对路径和文件名
    make_dir(save_path['absolute_path'])
    basic_path = save_path['absolute_path']
    print(basic_path)
    model_save_name = ''.join([basic_path, '/model_', str(settings_dict['current_exp_time']), '.pkl'])
    # 2. 获取模型
    print('获取模型：%s ' % model.get_model_name())
    model.init()
    if utils['parameter_optimization']:
        # 3. GridSearchCV参数优化
        print('GridSearchCV参数优化: ')
        best_params = model.parameter_optimization(x_train, y_train)
        print('优化后的参数: ', best_params)
        # 4.设置参数
        print('设置最优参数: ')
        model.set_params(best_params)
    else:
        print('使用默认参数: ')
    # 5. 模型训练
    print('模型训练: ')
    model.train(x_train, y_train)
    # 6. 模型测试
    print('模型测试: ')
    # if model.get_model_name() in ['LDA', 'RF']:
    #     pre_y_train, pre_y_test, pre_y_train_proba, pre_y_test_proba = model.predict(x_test, y_test)
    # else:
    #     pre_y_train, pre_y_test = model.predict(x_test, y_test)
    pre_y_train, pre_y_test = model.predict(x_test)
    # 7. 保存模型
    if utils['save_model']:
        print('保存最后一次训练后的模型：')
        model.save(model_save_name)
    # 8. 保存测试结果
    test_results_utils = GetTestResults(utils['test_metrics'], y_test, pre_y_test, decimal=5)
    test_metrics = test_results_utils.calculate()
    test_metrics_save_name = ''.join([basic_path, '/test_metrics.csv'])
    pre_results_save_name = ''.join([basic_path, '/predicted_results_', str(settings_dict['current_exp_time']), '.csv'])
    test_results_utils.save(settings_dict, test_metrics, test_metrics_save_name, pre_results_save_name)
    test_metrics_dict = dict(zip(utils['test_metrics'], test_metrics))
    print('测试结果：')
    for key, value in test_metrics_dict.items():
        print(f"{key}:  {value}.")

    # 9. 混淆矩阵操作
    if utils['confusion_matrix']['get_cm']:
        print('混淆矩阵：')
        cm_save_jpg_name = ''.join([basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '.jpg'])
        cm_save_csv_name = ''.join([basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '.xlsx'])
        plot_confusion_matrix = PlotConfusionMatrix(y_test, pre_y_test,
                                                    label_type=utils['confusion_matrix']['params']['label_type'],
                                                    show_type=utils['confusion_matrix']['params']['show_type'],
                                                    plot=utils['confusion_matrix']['params']['plot'],
                                                    save_fig=utils['confusion_matrix']['params']['save_fig'],
                                                    save_results=utils['confusion_matrix']['params']['save_results'],
                                                    cmap=utils['confusion_matrix']['params']['cmap'], )
        plot_confusion_matrix.get_confusion_matrix(cm_save_jpg_name, cm_save_csv_name)
