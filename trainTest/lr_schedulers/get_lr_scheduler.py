from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LrScheduler:
    def __init__(self, optimizer, scheduler_type, params, max_epoch):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.params = params
        self.epoch = max_epoch

    def get_scheduler(self):
        if self.scheduler_type == 'StepLR':
            scheduler = lr_scheduler.StepLR(self.optimizer,
                                            step_size=self.params['StepLR']['step_size'],
                                            gamma=self.params['StepLR']['gamma'])
        elif self.scheduler_type == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['MultiStepLR']['milestones'],
                                                 gamma=self.params['MultiStepLR']['gamma'])
        elif self.scheduler_type == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(self.optimizer,
                                                   gamma=self.params['ExponentialLR']['gamma'])
        elif self.scheduler_type == 'AutoWarmupLR':
            scheduler = AutoWarmupLR(self.optimizer, num_warm=self.params['AutoWarmupLR']['num_warm'])
        elif self.scheduler_type == 'GradualWarmupLR':
            num_warm = self.params['GradualWarmupLR']['total_epoch']
            basic_scheduler = lr_scheduler.MultiStepLR(self.optimizer,
                                                       milestones=[int(0.2*self.epoch)+num_warm,
                                                                   int(0.4*self.epoch)+num_warm,
                                                                   int(0.6*self.epoch)+num_warm,
                                                                   int(0.8*self.epoch)+num_warm],
                                                       gamma=self.params['MultiStepLR']['gamma'])

            scheduler = GradualWarmupScheduler(self.optimizer,
                                               multiplier=self.params['GradualWarmupLR']['multiplier'],
                                               total_epoch=num_warm,
                                               after_scheduler=basic_scheduler)
        elif self.scheduler_type == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                       mode=self.params['ReduceLROnPlateau']['mode'],
                                                       factor=self.params['ReduceLROnPlateau']['factor'],
                                                       verbose=self.params['ReduceLROnPlateau']['verbose'],
                                                       threshold=self.params['ReduceLROnPlateau']['threshold'],
                                                       min_lr=self.params['ReduceLROnPlateau']['min_lr'],
                                                       threshold_mode='rel',
                                                       cooldown=0,
                                                       eps=1e-8)
        else:
            raise ValueError('optimizer_type must be one of [StepLR, MultiStepLR, ExponentialLR, AutoWarmupLR, '
                             'GradualWarmupLR, ReduceLROnPlateau]')

        return scheduler


# warm up衰减策略先从一个极低的学习率开始增加，增加到某一个值后再逐渐减少。
# 这样训练模型更加稳定，因为在刚开始时模型的参数都是随机初始化的，此时如果学习率应该取小一点，这样就不会使模型一下子跑偏。
# 随着训练的增加，逐渐的模型对数据比较熟悉，此时可以增大学习率加快收敛速度。
# 最后随着参数逐渐收敛，在学习率增大到某个数值后开始衰减。

class AutoWarmupLR:
    def __init__(self, optimizer, num_warm) -> None:
        self.optimizer = optimizer
        self.num_warm = num_warm
        self.lr = [group['lr'] for group in self.optimizer.param_groups]
        self.num_step = 0

    def __compute(self, lr) -> float:
        return lr * min(self.num_step ** (-0.5), self.num_step * self.num_warm ** (-1.5))

    def step(self) -> None:
        self.num_step += 1
        lr = [self.__compute(lr) for lr in self.lr]
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lr[i]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
