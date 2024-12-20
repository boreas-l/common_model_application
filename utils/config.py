# ------ coding : utf-8 ------
# @FileName     : config.py
# @Author       : lxc
# @Time         : 2024/8/20 11:28

import copy
import yaml
import os
import argparse


class AttrDict(dict):
    """获取字典相关字段属性"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


def create_attr_dict(yaml_config):
    """遍历config文件键值对，并生成dict对象"""
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def override(dl, ks, v):
    """
    更新config文件键值对
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                print('A new field ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                print("A new Series field ({}) detected!".format(ks[0], dl))
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    在给config文件内的键值对中，指定新的值时进行更新覆盖
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)
    return config


def get_config(fname, overrides=None):
    """
    读取配置文件
    """
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    override_config(config, overrides)

    return config


def parse_args():
    parser = argparse.ArgumentParser("通用模型运行参数")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.config = "../configs/ResNet18.yaml"
    config_content = get_config(args.config, args.override)
    print(f">>>config: {config_content}")
