import torch.nn as nn
from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer
from .logitnorm_trainer import LogitNormTrainer

class SPCPLogitNormTrainer(LogitNormTrainer):
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:
        super(SPCPLogitNormTrainer, self).__init__(net, train_loader, config)
        self.net.set_alpha(config.trainer.trainer_args.alpha)
        self.net.set_p(config.trainer.trainer_args.p)
        self.net.set_T(config.trainer.trainer_args.T)
