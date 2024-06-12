import torch.optim as optim


class Optimizer:
    def __init__(self, model, optimizer_type='Adam', lr=0.001):
        self.model = model
        self.optimizer_type = optimizer_type
        self.lr = lr

    def get_optimizer(self):
        if self.optimizer_type == 'Adam':
            print('使用Adam优化器，初始学习率: ', self.lr)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                   amsgrad=False)
        elif self.optimizer_type == 'RMSprop':
            print('使用RMSprop优化器，初始学习率: ', self.lr)
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0,
                                      momentum=0, centered=False)
        else:
            raise ValueError('optimizer_type must be one of [Adam, RMSprop]')

        return optimizer
