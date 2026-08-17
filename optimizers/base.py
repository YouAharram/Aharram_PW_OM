from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class OptimizerInterface(ABC):

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ):
        pass

    @abstractmethod
    def get_metrics(self):
        pass

    @abstractmethod
    def get_lr(self):
        pass
