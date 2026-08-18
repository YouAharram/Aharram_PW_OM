import torch
import torch.nn as nn

from .base import OptimizerInterface


class SAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        eta=0.1,
        rho=0.1,
    ):
        params = list(params)

        super().__init__(params, {})

        self.params = params

        self.eta = eta
        self.rho = rho

        self.reset_epoch()

        self.state["n_forwards"] = 0
        self.state["n_backwards"] = 0

    def step(self, closure_mini_batch):

        params_current = [
            p.detach().clone()
            for p in self.params
        ]

        # First forward + backward
        loss = closure_mini_batch()

        self.state["n_forwards"] += 1
        self.state["n_backwards"] += 1

        grad_norm = torch.sqrt(
            sum(
                torch.sum(p.grad ** 2)
                for p in self.params
                if p.grad is not None
            )
        )

        grad_norm = grad_norm.clamp_min(1e-12)

        # Move to worst point
        self.update_to_worst_point(
            params_current,
            self.rho / grad_norm,
        )

        # Second forward + backward
        self.zero_grad()

        loss_worst_point = closure_mini_batch()

        self.state["n_forwards"] += 1
        self.state["n_backwards"] += 1

        # Update parameters
        self.sam_update(
            params_current,
            self.eta,
        )

        self.state["all_loss"].append(
            loss.item()
        )

        return loss

    def update_to_worst_point(
        self,
        params_current,
        step,
    ):
        with torch.no_grad():

            for p, p_curr in zip(
                self.params,
                params_current,
            ):
                if p.grad is not None:
                    p.copy_(
                        p_curr + step * p.grad
                    )

    def sam_update(
        self,
        params_current,
        step,
    ):
        with torch.no_grad():

            for p, p_curr in zip(
                self.params,
                params_current,
            ):
                if p.grad is not None:
                    p.copy_(
                        p_curr - step * p.grad
                    )

    def reset_epoch(self):

        self.state["all_step"] = []
        self.state["all_loss"] = []
        self.state["backtracks"] = 0

class SAMOptimizer(OptimizerInterface):

    def __init__(
        self,
        model: nn.Module,
        eta: float,
        rho: float,
    ):
        self.optimizer = SAM(
            model.parameters(),
            eta=eta,
            rho=rho,
        )

    def train_step(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ):
        self.optimizer.zero_grad()

        def closure():
            self.optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            return loss

        loss = self.optimizer.step(closure)

        return loss

    def get_metrics(self):
        return {
            "n_forwards": self.optimizer.state["n_forwards"],
            "n_backwards": self.optimizer.state["n_backwards"],
        }

    def get_lr(self):
        return self.optimizer.eta
