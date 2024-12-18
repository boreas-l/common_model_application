# ------ coding : utf-8 ------
# @FileName     : logger.py
# @Author       : lxc
# @Time         : 2024/8/20 16:34

import logging
import os

_logger = None
def init_logger(log_file='train.log'):
    """初始化一个日志对象"""
    global _logger
    #
    log_file_path = os.path.dirname(log_file)
    os.makedirs(log_file_path, exist_ok=True)
    # 创建一个logger
    logger = logging.getLogger('model_training')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    logger = init_logger()
    logger.info('This is a test log.')
    logger.error('has a error')