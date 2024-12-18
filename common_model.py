# ------ coding : utf-8 ------
# @FileName     : common_model.py
# @Author       : lxc
# @Time         : 2024/8/20 16:03

import platform
import torch
from utils.tools import setup_seed
from utils.logger import init_logger
from utils.train_main import train_epoch
from utils.data_loader import build_dataloader
from network.optimizer.build_optimizer import build_optimizer
from network.loss.build_loss import build_loss
from network.metric.build_metric import build_metric
from network import build_model


class CommonModel:
    def __init__(self, config):
        self.config = config
        print(f"config: {self.config}")
        # 固定随机种子，保证训练结果可复现
        setup_seed(132)
        # 初始化日志对象
        self.logger = init_logger(log_file=f"{self.config['Global']['output_dir']}/train.log")
        # 设置device
        if self.config['Global']['device'] == 'gpu':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        # gradient accumulation
        self.update_freq = self.config["Global"].get("update_freq", 1)
        # 初始化模型
        self.model = build_model(self.config)
        self.model.to(self.device)
        # 初始化dataloader
        self.dataloader = build_dataloader(self.config['DataLoader'], mode='Train')
        # 初始化optimizer
        self.optimizer, self.scheduler = self.init_optimier(self.model, self.config['Optimizer'], self.config['Scheduler'])
        # 初始化loss
        self.loss = build_loss(self.config['Loss'])
        # 初始化metric
        self.metric = build_metric(self.config['Metric'])


    def init_optimier(self, model, optim_config, scheduler_config):
        optimizer, scheduler = build_optimizer(model, optim_config, scheduler_config)

        return optimizer, scheduler

    def train(self):
        best_metric = {
            "metric": -1.0,
            "epoch": 0,
        }
        max_iter = len(self.dataloader) - 1 if platform.system() == "Windows" else len(self.dataloader)
        max_iter = max_iter // self.update_freq * self.update_freq
        train_config = {
            "max_iter": max_iter,
            "global_step": 0,
            "device": self.device,
            "epoch": 0,
            "update_freq": self.update_freq,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "metric": self.metric,
        }
        for epoch in range(best_metric['epoch']+1, self.config["Global"]["epochs"]+1):
            # train for one epoch
            train_epoch(self.model, self.dataloader, train_config, self.loss, self.logger)
            print("train config: ", train_config)
            break


if __name__ == '__main__':
    from utils.config import parse_args, get_config

    args = parse_args()
    args.config = "./configs/ResNet18.yaml"
    config_content = get_config(args.config, args.override)
    print(f">>>config: {config_content}")
    common_model_pipeline = CommonModel(config_content)
    common_model_pipeline.train()
    print("done")
