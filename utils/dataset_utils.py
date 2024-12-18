# ------ coding : utf-8 ------
# @FileName     : dataset_utils.py
# @Author       : lxc
# @Time         : 2024/8/15 16:58

from utils.transforms import transforms_list


def create_operators(params):
    """
    基于传参构建预处理操作
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator.keys())[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if op_name == 'ToTensor':
            op = transforms_list[op_name]()
        else:
            op = transforms_list[op_name](**param)
        ops.append(op)

    return ops


def do_transforms(data, ops=None):
    """执行图像处理"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
    return data
