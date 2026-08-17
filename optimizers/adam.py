import torch
import torch.nn as nn

from .base import OptimizerInterface


class AdamOptimizer(OptimizerInterface):

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float = 0.0,
    ):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.n_forwards = 0
        self.n_backwards = 0

    def train_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ):
        self.optimizer.zero_grad()

        outputs = model(images)
        self.n_forwards += 1

        loss = criterion(outputs, labels)

        loss.backward()
        self.n_backwards += 1

        self.optimizer.step()

        return loss, outputs

    def get_metrics(self):
        return {
            "n_forwards": self.n_forwards,
            "n_backwards": self.n_backwards,
        }
