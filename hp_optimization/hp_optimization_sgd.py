import torch
import torch.nn as nn

from datasets import get_cifar10
from models import ResNet18
from optimizers import create_sgd
from training import train


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

learning_rates = [0.01, 0.05, 0.1, 0.2]

train_loader, val_loader, _ = get_cifar10(
    batch_size=128
)

results = {}

for lr in learning_rates:

    print("\n" + "=" * 50)
    print(f"Testing SGD with learning rate = {lr}")
    print("=" * 50)

    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = create_sgd(
        model=model,
        lr=lr,
        momentum=0.9,
        epochs=10,
    )

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
        scheduler=scheduler,
    )

    results[lr] = history["best_val_accuracy"]

print("\nResults:")
print("-" * 30)

for lr, accuracy in results.items():
    print(
        f"LR: {lr:<6} "
        f"Best Val Accuracy: {accuracy:.4f}"
    )
