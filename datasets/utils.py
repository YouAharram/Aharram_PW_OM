import torch
from torch.utils.data import random_split

def train_val_split(dataset, val_ratio: float, seed: int = 42):
    train_size = len(dataset)

    if(val_ratio < 0 or val_ratio > 1):
        raise ValueError("val_ratio should be beatween 0 and 1")

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator = generator,
            )

    return train_dataset, val_dataset
