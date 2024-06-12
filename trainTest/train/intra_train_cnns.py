import torch
import numpy as np
import pandas as pd
# 导入模型可视化和训练过程可视化库
import hiddenlayer as hl
from tqdm import tqdm
# 导入小工具
from utils.common_utils import make_dir
# 导入模型分类性能评价工具
from trainTest.metrics.sklearn_version import get_accuracy
from trainTest.metrics.get_test_metrics import GetTestResults, PlotConfusionMatrix
from trainTest.visualization.tsne import TsneVisualizationCNNs
# 导入模型训练过程的优化工具
from trainTest.losses.get_loss import ClassificationLoss
from trainTest.optimizers.get_optimizer import Optimizer
from trainTest.earlyStopping.early_stopping import EarlyStopping
from trainTest.lr_schedulers.get_lr_scheduler import LrScheduler
# 隐马尔可夫模型，用于步态识别任务
from models.hmm.hmm_tools_simple import HMMGaitClassification


def train_test_intra_model(settings_dict, model, train_loader, valid_loader, test_loader, device, save_path, callbacks,
                           utils, true_labels):
    early_stopping, scheduler, history_save_pic_name = None, None, None
    # 1. 模型、测试结果保存绝对路径和文件名
    make_dir(save_path['absolute_path'])
    basic_path = ''.join(['../results/Intra-Subject/4-GNNs-TRGRatioTest/', save_path['relative_path']])
    # basic_path = save_path['absolute_path']
    model_save_name = ''.join([basic_path, '/model_', str(settings_dict['current_exp_time']), '.pt'])
    history_save_pkl_name = ''.join([basic_path, '/history_', str(settings_dict['current_exp_time']), '.pkl'])
    history_save_csv_name = ''.join([basic_path, '/history_', str(settings_dict['current_exp_time']), '.csv'])

    # 2. callbacks
    optimizer = Optimizer(model, optimizer_type=callbacks['optimizer'], lr=callbacks['initial_lr']).get_optimizer()
    if callbacks['lr_scheduler']['scheduler_type'] == 'None':
        pass
    else:
        print('使用学习率调度器：', callbacks['lr_scheduler']['scheduler_type'])
        scheduler = LrScheduler(optimizer, callbacks['lr_scheduler']['scheduler_type'],
                                callbacks['lr_scheduler']['params'], callbacks['epoch']).get_scheduler()

    if callbacks['early_stopping']['use_es']:
        print('使用早停：')
        early_stopping = EarlyStopping(patience=callbacks['early_stopping']['params']['patience'],
                                       verbose=callbacks['early_stopping']['params']['verbose'],
                                       delta=callbacks['early_stopping']['params']['delta'],
                                       path=model_save_name)

    # 3. train_test_utils
    history = hl.History()
    progress_bar, canvas = None, None
    if utils['use_tqdm']:
        progress_bar = tqdm(total=callbacks['epoch'])
    if utils['train_plot']:
        history_save_pic_name = ''.join([basic_path, '/history_', str(settings_dict['current_exp_time']), '.jpg'])
        canvas = hl.Canvas()

    for e in range(1, 1 + callbacks['epoch']):
        train_predict_labels, train_true_labels, train_loss = [], [], 0.0
        criterion = ClassificationLoss(device, loss_type=callbacks['criterion'], callbacks=callbacks, utils=utils,
                                       epoch=e, class_weights=callbacks['weights']).get_criterion()
        # 4. 训练循环
        model.train()
        for data, feature, target in train_loader:
            data = data.to(device=device)
            feature = feature.to(device=device)
            label = target
            target = target.to(device=device).view(-1)
            predict = model(data, feature)
            loss = criterion(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
            train_predict_labels.append(predict_index)
            train_true_labels.append(label)
            train_loss += loss.detach().to(device='cpu').item()  # 累积训练集损失
        train_predict_labels = torch.cat(train_predict_labels, dim=0).view(-1).numpy()
        train_true_labels = torch.cat(train_true_labels, dim=0).view(-1).numpy()
        acc_train_epoch = get_accuracy(train_true_labels, train_predict_labels, decimal=5)
        loss_train_epoch = np.round(train_loss / len(train_loader), 5)

        # 5. 验证循环
        if utils['model_eval']['valid']:
            model.eval()
        valid_predict_labels, valid_true_labels, valid_loss = [], [], 0.0
        with torch.no_grad():
            for data, feature, target in valid_loader:
                data = data.to(device=device)
                feature = feature.to(device=device)
                label = target
                target = target.to(device=device).view(-1)
                predict = model(data, feature)
                loss = criterion(predict, target)
                _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
                valid_predict_labels.append(predict_index)
                valid_true_labels.append(label)
                valid_loss += loss.detach().to(device='cpu').item()  # 累积验证集损失
        valid_predict_labels = torch.cat(valid_predict_labels, dim=0).view(-1).numpy()
        valid_true_labels = torch.cat(valid_true_labels, dim=0).view(-1).numpy()
        acc_valid_epoch = get_accuracy(valid_true_labels, valid_predict_labels, decimal=5)
        loss_valid_epoch = np.round(valid_loss / len(valid_loader), 5)
        # 验证结束后才更新学习率
        if callbacks['lr_scheduler']['scheduler_type'] in ['StepLR', 'MultiStepLR', 'AutoWarmupLR', 'GradualWarmupLR',
                                                           'ExponentialLR']:
            scheduler.step()
        else:
            #  callbacks['lr_scheduler']['scheduler_type'] == 'ReduceLROnPlateau'
            scheduler.step(acc_valid_epoch / 100)

        # 6. 绘图或打印
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        history.log(e, train_acc=acc_train_epoch, valid_acc=acc_valid_epoch, train_loss=loss_train_epoch,
                    valid_loss=loss_valid_epoch, lr=lr)
        if utils['train_plot']:
            with canvas:
                canvas.draw_plot([history["train_acc"], history["valid_acc"]],
                                 labels=["Train accuracy", "Valid accuracy"])
                canvas.draw_plot([history["train_loss"], history["valid_loss"]],
                                 labels=["Train loss", "Valid loss"])
                canvas.draw_plot([history["lr"]], labels=["lr"])

        if utils['use_tqdm']:
            progress_bar.update(1)
            progress_bar.set_description(
                f"train_acc: {acc_train_epoch:.3f}, valid_acc: {acc_valid_epoch:.3f}, train_loss: {loss_train_epoch:.3f}, valid_loss: {loss_valid_epoch:.3f}")
        else:
            if e == 1 or e % utils['print_interval'] == 0:
                history.progress()

        if callbacks['early_stopping']['use_es']:
            # 判断早停
            early_stopping(loss_valid_epoch, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping，保存模型：")
                break

    # 7. 如果不使用早停中的模型保存，则在模型训练结束后保存模型
    if not callbacks['early_stopping']['use_es']:
        print('保存最后一次训练后的模型：')
        torch.save(model, model_save_name)

    # 8. 保存模型训练过程：csv / pkl,  jpg
    print('保存模型训练过程：')
    history.save(history_save_pkl_name)
    # 加载pkl文件
    # h1 = hl.History()
    # h1.load(history_save_pic_name)
    # print(history.history)
    history_metrics = pd.DataFrame(history.history).T
    history_metrics.to_csv(history_save_csv_name, index=True)
    if utils['train_plot']:
        canvas.save(history_save_pic_name)

    # 9. 整个训练轮数结束后才进行测试
    # 结束训练后停止进度条
    if utils['use_tqdm']:
        progress_bar.close()
    # 这里需要重新加载已经保存的模型，因为如果采用了早停，则保存的是验证集上最好的模型，如果不加载默认是最后一次训练结束后的模型
    model = torch.load(model_save_name)
    test_predict_labels, test_true_labels, test_loss = [], [], 0.0
    if utils['model_eval']['test']:
        model.eval()
    with torch.no_grad():
        for data, feature, target in test_loader:
            data = data.to(device=device)
            feature = feature.to(device=device)
            label = target
            # target = target.to(device=device).view(-1)
            predict = model(data, feature)
            # loss = criterion(predict, target)
            _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
            test_predict_labels.append(predict_index)
            test_true_labels.append(label)
            # test_loss += loss.detach().to(device='cpu').item()  # 累积测试集损失

    test_predict_labels = torch.cat(test_predict_labels, dim=0).view(-1).numpy()
    test_true_labels = torch.cat(test_true_labels, dim=0).view(-1).numpy()

    # 10. 保存测试结果
    test_results_utils = GetTestResults(utils['test_metrics'], test_true_labels, test_predict_labels, decimal=5)
    test_metrics = test_results_utils.calculate()
    test_metrics_save_name = ''.join([basic_path, '/test_metrics.csv'])
    pre_results_save_name = ''.join([basic_path, '/predicted_results_', str(settings_dict['current_exp_time']), '.csv'])
    test_results_utils.save(settings_dict, test_metrics, test_metrics_save_name, pre_results_save_name)
    test_metrics_dict = dict(zip(utils['test_metrics'], test_metrics))
    print('测试结果：')
    for key, value in test_metrics_dict.items():
        print(f"{key}:  {value}.")

    # 11. 混淆矩阵操作
    if utils['confusion_matrix']['get_cm']:
        print('混淆矩阵：')
        cm_save_jpg_name = ''.join([basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '.jpg'])
        cm_save_csv_name = ''.join([basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '.xlsx'])
        plot_confusion_matrix = PlotConfusionMatrix(test_true_labels, test_predict_labels,
                                                    label_type=utils['confusion_matrix']['params']['label_type'],
                                                    show_type=utils['confusion_matrix']['params']['show_type'],
                                                    plot=utils['confusion_matrix']['params']['plot'],
                                                    save_fig=utils['confusion_matrix']['params']['save_fig'],
                                                    save_results=utils['confusion_matrix']['params']['save_results'],
                                                    cmap=utils['confusion_matrix']['params']['cmap'], )
        plot_confusion_matrix.get_confusion_matrix(cm_save_jpg_name, cm_save_csv_name)

    # 12. 步态分类任务的hmm_tools
    if utils['hmm_tools']['gait'] == 'gait' and utils['hmm_tools']['use_hmm']:
        hmm_predict_labels = HMMGaitClassification(model, train_loader, test_loader, true_labels,
                                                   utils['hmm_tools']['gait_dataset_divide_mode'],
                                                   num_classes=len(utils['confusion_matrix']['params']['label_type']),
                                                   ).init()
        # 12.1. 保存测试结果
        print(test_true_labels.shape)
        hmm_results_utils = GetTestResults(utils['test_metrics'], test_true_labels, hmm_predict_labels, decimal=5)
        hmm_test_metrics = hmm_results_utils.calculate()
        hmm_metrics_save_name = ''.join([basic_path, '/test_metrics_hmm.csv'])
        hmm_pre_results_save_name = ''.join(
            [basic_path, '/predicted_results_hmm_', str(settings_dict['current_exp_time']), '.csv'])
        hmm_results_utils.save(settings_dict, hmm_test_metrics, hmm_metrics_save_name, hmm_pre_results_save_name)
        hmm_metrics_dict = dict(zip(utils['test_metrics'], hmm_test_metrics))
        print('HMM修正后的步态预测结果：')
        for key, value in hmm_metrics_dict.items():
            print(f"{key}:  {value}.")

        # 12.2. 混淆矩阵操作
        if utils['confusion_matrix']['get_cm']:
            print('HMM修正后的混淆矩阵：')
            cm_save_jpg_name = ''.join(
                [basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '_hmm.jpg'])
            cm_save_csv_name = ''.join(
                [basic_path, '/confusion_matrix_', str(settings_dict['current_exp_time']), '_hmm.xlsx'])
            plot_confusion_matrix = PlotConfusionMatrix(test_true_labels, hmm_predict_labels,
                                                        label_type=utils['confusion_matrix']['params']['label_type'],
                                                        show_type=utils['confusion_matrix']['params']['show_type'],
                                                        plot=utils['confusion_matrix']['params']['plot'],
                                                        save_fig=utils['confusion_matrix']['params']['save_fig'],
                                                        save_results=utils['confusion_matrix']['params'][
                                                            'save_results'],
                                                        cmap=utils['confusion_matrix']['params']['cmap'], )
            plot_confusion_matrix.get_confusion_matrix(cm_save_jpg_name, cm_save_csv_name)

    # 13.T-SNE操作
    if utils['tsne_visualization']['get_tsne']:
        print('tsne可视化：%s, %s' % (
            utils['tsne_visualization']['params']['show_type'], utils['tsne_visualization']['params']['feature_type']))
        if utils['tsne_visualization']['params']['show_type'] in ['train_set', 'test_set']:
            tsne_result_name = ''.join([basic_path, '/tsne_', utils['tsne_visualization']['params']['show_type'], '_',
                                        utils['tsne_visualization']['params']['feature_type'], '_',
                                        str(settings_dict['current_exp_time']), '.csv'])
            tsne_fig_name = ''.join([basic_path, '/tsne_', utils['tsne_visualization']['params']['show_type'], '_',
                                     utils['tsne_visualization']['params']['feature_type'], '_',
                                     str(settings_dict['current_exp_time']), '.jpg'])
            data_loader = train_loader if utils['tsne_visualization']['params'][
                                              'show_type'] == 'train_set' else test_loader
            TsneVisualizationCNNs(model=model, data_loader=data_loader,
                                  feature_type=utils['tsne_visualization']['params']['feature_type'],
                                  branch=settings_dict['network_branch'],
                                  save_results=utils['tsne_visualization']['params']['save_results'],
                                  save_fig=utils['tsne_visualization']['params']['save_fig'],
                                  result_name=tsne_result_name, fig_name=tsne_fig_name,
                                  num_classes=len(utils['confusion_matrix']['params']['label_type'])
                                  ).init()
        elif utils['tsne_visualization']['params']['show_type'] == 'all':
            for dataset in ['train_set', 'test_set']:
                tsne_result_name = ''.join(
                    [basic_path, '/tsne_', dataset, '_', utils['tsne_visualization']['params']['feature_type'], '_',
                     str(settings_dict['current_exp_time']), '.csv'])
                tsne_fig_name = ''.join(
                    [basic_path, '/tsne_', dataset, '_', utils['tsne_visualization']['params']['feature_type'], '_',
                     str(settings_dict['current_exp_time']), '.jpg'])
                data_loader = train_loader if dataset == 'train_set' else test_loader
                TsneVisualizationCNNs(model=model, data_loader=data_loader,
                                      feature_type=utils['tsne_visualization']['params']['feature_type'],
                                      branch=settings_dict['network_branch'],
                                      save_results=utils['tsne_visualization']['params']['save_results'],
                                      save_fig=utils['tsne_visualization']['params']['save_fig'],
                                      result_name=tsne_result_name, fig_name=tsne_fig_name,
                                      num_classes=len(utils['confusion_matrix']['params']['label_type'])
                                      ).init()
        else:
            raise ValueError('show_type must be one of "all" or "train_set"or "test_set"')
