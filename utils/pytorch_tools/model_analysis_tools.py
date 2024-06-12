import os
import torch
from torchviz import make_dot
from torchsummary import summary
from thop import profile
from utils.common_utils import printlog, make_dir
from fvcore.nn import parameter_count_table
# import hiddenlayer as h


"""
1. 针对单分支输入的网络的模型工具包，包括四个功能：
1）测试模型输出维度; 
2）计算模型复杂度: 时间复杂度：macs, madds; 空间复杂度：params size, 存储参数所需的memory；
3）模型可视化，打印每层网络的输出尺寸；
4）模型可视化，保存模型结构为.pdf或.jpg文件， 默认.pdf类型。
"""


class OneBranchModelTools:
    def __init__(self, target_model, input_size, batch_size, device):
        self.model = target_model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.input_data = torch.randn(batch_size, input_size[0], input_size[1], input_size[2]).to(self.device)
        self.model = self.model.to(self.device)
        # self.model.cuda()

    def summary_model(self, method='summary'):
        if method == 'summary':
            printlog(info='使用torchsummary打印模型每个网络层的形状', time=True, line_break=True)
            summary(self.model, input_data=self.input_size, depth=8)
        elif method == 'fvcore':
            printlog(info='使用fvcore.nn打印模型每个网络层的形状', time=True, line_break=True)
            parameter_table = parameter_count_table(self.model, max_depth=3)
            print(parameter_table)
        else:
            raise ValueError('method must be summary or fvcore')

    def plot_model(self, model_name, save_format='pdf', show=True, verbose=False):
        # 使用 torchviz 生成模型结构图
        printlog(info='使用 torchviz 生成模型结构图', time=True, line_break=True)
        dot = make_dot(self.model(self.input_data), params=dict(self.model.named_parameters()), show_attrs=verbose, show_saved=verbose)
        # 将结构图保存为 pdf图片
        file_dir = os.path.join(os.getcwd(), 'modelVisualization')
        make_dir(file_dir)
        file_name = os.path.join(file_dir, model_name)
        dot.format = save_format
        if show:
            dot.view(filename=file_name, directory=file_dir, cleanup=True)
        else:
            dot.render(filename=file_name, directory=file_dir, cleanup=True)
        # vis_graph = h.build_graph(self.model, self.input_data)  # 获取绘制图像的对象
        # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
        # vis_graph.save("./demo1.png")  # 保存图像的路径

    def calculate_complexity(self, verbose=True, print_log=True):
        # 模型复杂度计算
        printlog(info='计算模型复杂度', time=True, line_break=True)
        # self.model = self.model.to(self.device)
        # self.input_data = self.input_data.to(self.device)
        macs, params = profile(self.model, inputs=(self.input_data, ), verbose=verbose, report_missing=verbose)
        if print_log:
            print('macs:', macs, 'params:', params)
        return macs, params

    def test_output_shape(self):
        printlog(info='测试模型输出维度', time=True, line_break=True)
        output = self.model(self.input_data)
        print('output.shape', output.shape)


"""
2. 针对双分支输入的网络的模型工具包。
"""


class DualBranchModelTools:
    def __init__(self, target_model, input_size, batch_size, device):
        self.model = target_model
        self.input_size, self.input_size_0, self.input_size_1 = input_size, input_size[0], input_size[1]
        self.batch_size = batch_size
        self.device = device
        self.input_data_0 = torch.randn(batch_size, self.input_size_0[0], self.input_size_0[1], self.input_size_0[2]).to(self.device)
        self.input_data_1 = torch.randn(batch_size, self.input_size_1[0], self.input_size_1[1], self.input_size_1[2]).to(self.device)
        self.model = self.model.to(self.device)
        # self.model.cuda()

    def summary_model(self, method='summary'):
        if method == 'summary':
            printlog(info='使用torchsummary打印模型每个网络层的形状', time=True, line_break=True)
            summary(self.model, input_data=self.input_size, depth=8)
        elif method == 'fvcore':
            printlog(info='使用fvcore.nn', time=True, line_break=True)
            parameter_table = parameter_count_table(self.model, max_depth=3)
            print(parameter_table)
        else:
            raise ValueError('method must be summary or fvcore')

    def plot_model(self, model_name, save_format='pdf', show=True, verbose=False):
        # 使用 torchviz 生成模型结构图
        printlog(info='使用 torchviz 生成模型结构图', time=True, line_break=True)
        dot = make_dot(self.model(self.input_data_0, self.input_data_1), params=dict(self.model.named_parameters()), show_attrs=verbose, show_saved=verbose)
        # 保存模型结构图：默认.pdf类型，可以改为.jpg等图片格式
        file_dir = os.path.join(os.getcwd(), 'modelVisualization')
        make_dir(file_dir)
        file_name = os.path.join(file_dir, model_name)
        dot.format = save_format
        if show:
            dot.view(filename=file_name, directory=file_dir, cleanup=True)
        else:
            dot.render(filename=file_name, directory=file_dir, cleanup=True)

    def calculate_complexity(self, verbose=True, print_log=True):
        # 模型复杂度计算
        printlog(info='计算模型复杂度', time=True, line_break=True)
        # self.model = self.model.to(self.device)
        # self.input_data = self.input_data.to(self.device)
        macs, params = profile(self.model, inputs=(self.input_data_0, self.input_data_1), verbose=verbose, report_missing=verbose)
        if print_log:
            print('macs:', macs, 'params:', params)
        return macs, params

    def test_output_shape(self):
        printlog(info='测试模型输出维度', time=True, line_break=True)
        output = self.model(self.input_data_0, self.input_data_1)
        print('output.shape', output.shape)
