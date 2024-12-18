# ------ coding : utf-8 ------
# @FileName     : tools.py
# @Author       : lxc
# @Time         : 2024/8/20 16:53

"""相关工具函数"""
import torch
import torch.optim as optim
import numpy as np
import random


def setup_seed(seed=0):
    # 为Python的random模块设置随机种子
    random.seed(seed)

    # 为numpy设置随机种子
    np.random.seed(seed)

    # 为CPU设置随机种子
    torch.manual_seed(seed)

    # 为当前GPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU

        # 这两行可选，但建议加上，特别是如果你希望得到可重复的结果
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_optimizer(optimizer_name, model, optimizer_params):
    # 定义一个字典，将优化器名称映射到对应的优化器类
    optimizer_dict = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'rmsprop': optim.RMSprop,
        'momentum': optim.Momentum,
    }

    # 检查优化器名称是否存在于字典中
    if optimizer_name not in optimizer_dict:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

    # 获取对应的优化器类
    optimizer_class = optimizer_dict[optimizer_name]

    # 创建优化器实例
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    return optimizer

if __name__ == '__main__':
    import yaml

    config_file_path = "../configs/ResNet18.yaml"
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        optimizer_content = config['Optimizer']
        print(optimizer_content)
