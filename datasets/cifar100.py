from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import train_val_split


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
        ),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset, val_dataset = train_val_split(
        train_dataset,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
