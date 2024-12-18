# ------ coding : utf-8 ------
# @FileName     : build_loss.py
# @Author       : lxc
# @Time         : 2024/10/8 16:53

import torch.nn as nn

loss_dict = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
}


def build_loss(config):
    """
    构建损失函数
    :param config:
    :param class_num
    :return:
    """
    loss_name = config.pop('name')
    loss_func = loss_dict[loss_name]
    # 如果配置文件中，有weight关键词，则判断是否等于
    loss = loss_func(**config)

    return loss
