from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import train_val_split


SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def get_svhn(
    data_dir: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=SVHN_MEAN,
            std=SVHN_STD,
        ),
    ])

    train_dataset = datasets.SVHN(
        root=data_dir,
        split="train",
        download=True,
        transform=transform,
    )

    test_dataset = datasets.SVHN(
        root=data_dir,
        split="test",
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
