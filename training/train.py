import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluate import evaluate


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        loss = optimizer.train_step(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
        )

        with torch.no_grad():
            outputs = model(images)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total,
    )

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
):
    history = {
        "model": {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_val_accuracy": 0.0,
        },
        "optimizer": {
            "n_forwards": 0,
            "n_backwards": 0,
            "training_time": 0.0,
        },
    }

    best_val_accuracy = 0.0
    best_state_dict = None

    start_time = time.perf_counter()

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Model metrics
        history["model"]["train_loss"].append(
            train_loss
        )

        history["model"]["train_accuracy"].append(
            train_accuracy
        )

        history["model"]["val_loss"].append(
            val_loss
        )

        history["model"]["val_accuracy"].append(
            val_accuracy
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state_dict = copy.deepcopy(
                model.state_dict()
            )

        history["model"]["best_val_accuracy"] = (
            best_val_accuracy
        )

        # Optimizer metrics
        optimizer_metrics = optimizer.get_metrics()

        history["optimizer"].update(
            optimizer_metrics
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

    history["optimizer"]["training_time"] = (
        time.perf_counter() - start_time
    )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return history
