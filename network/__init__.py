# ------ coding : utf-8 ------
# @FileName     : __init__.py
# @Author       : lxc
# @Time         : 2024/8/15 10:33

import copy
from network.backbones import *
from network.layers import *

# 模型列表
model_dict = {
    'ResNet18': resnet_18,
    'ResNet34': resnet_34,
    'ResNet50': resnet_50,
    'ResNet101': resnet_101
}

def build_model(config):
    arch_config = copy.deepcopy(config["Arch"])
    model_name = arch_config.pop("name")
    assert model_name in model_dict.keys(), f"请检查配置文件的模型名称是否有误，当前名称为-{model_name}"
    model = model_dict[model_name](**arch_config)

    return model


if __name__ == '__main__':
    from utils.config import get_config

    config = get_config("../configs/ResNet18.yaml")
    print("config: ", config)
    model = build_model(config)
    print("model: ", model)
