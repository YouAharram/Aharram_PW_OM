import torch


def create_sgd(
    model,
    lr: float,
    momentum: float = 0.9,
    epochs: int = 100,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay = 0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    return optimizer, scheduler


def create_adam(
    model,
    lr: float,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay = 0.0,
    )

    return optimizer
