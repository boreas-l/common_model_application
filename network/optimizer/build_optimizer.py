# ------ coding : utf-8 ------
# @FileName     : build_optimizer.py
# @Author       : lxc
# @Time         : 2024/9/10 16:02

import torch.optim as optim

optim_dict = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
    'Adadelta': optim.Adadelta,
    'Adagrad': optim.Adagrad,
    'Adamax': optim.Adamax,
    'ASGD': optim.ASGD,
    'RMSprop': optim.RMSprop,
    'Rprop': optim.Rprop,
    'SparseAdam': optim.SparseAdam
}

lr_schedule_dict = {
    'LambdaLR': optim.lr_scheduler.LambdaLR,
    'StepLR': optim.lr_scheduler.StepLR
}


def build_optimizer(model, optim_cfg, scheduler_cfg):
    """
    构建优化器
    :param model:
    :param optimizer_cfg:
    :param scheduler_cfg:
    :return:
    """
    # 构建优化器
    optim_name = optim_cfg.pop("name")
    optimizer_func = optim_dict[optim_name]
    optimizer = optimizer_func(model.parameters(), **optim_cfg)
    # 构建学习率调度器
    scheduler_name = scheduler_cfg.pop("name")
    scheduler_func = lr_schedule_dict[scheduler_name]
    scheduler = scheduler_func(optimizer, **scheduler_cfg)

    return optimizer, scheduler
