import torch
import torch.nn as nn

from datasets import get_cifar10
from models import ResNet18
from optimizers import create_adam
from training import train


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]

train_loader, val_loader, _ = get_cifar10(
    batch_size=128
)

results = {}

for lr in learning_rates:

    print("\n" + "=" * 50)
    print(f"Testing Adam with learning rate = {lr}")
    print("=" * 50)

    model = ResNet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_adam(
        model=model,
        lr=lr,
    )

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=10,
    )

    results[lr] = history["best_val_accuracy"]

print("\nResults:")
print("-" * 30)

for lr, accuracy in results.items():
    print(
        f"LR: {lr:<8} "
        f"Best Val Accuracy: {accuracy:.4f}"
    )
