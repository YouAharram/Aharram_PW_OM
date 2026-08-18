import torch
import torch.nn as nn

from datasets import get_cifar10
from models import ResNet18
from optimizers import SGDOptimizer
from hp_optimization import hyperparameter_search

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


configs = [
    {"lr": 0.01},
    {"lr": 0.03},
    {"lr": 0.05},
    {"lr": 0.1},
]


def create_model():
    return ResNet18(num_classes=10)


def create_optimizer(model, lr):
    return SGDOptimizer(
        model=model,
        lr=lr,
        momentum=0.9,
    )


def create_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer,
        T_max=epochs,
    )


results = hyperparameter_search(
    dataset_fn=get_cifar10,
    model_fn=create_model,
    optimizer_fn=create_optimizer,
    hyperparameter_configs=configs,
    criterion_fn=nn.CrossEntropyLoss,
    device=device,
    epochs=10,
    batch_size=128,
    scheduler_fn=create_scheduler,
)


for config, result in results.items():

    print(
        config,
        result["best_val_accuracy"]
    )
