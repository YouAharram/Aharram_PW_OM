from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .utils import train_val_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_cifar10(
        data_dir: str = "./data",
        batch_size: int = 128,
        val_ratio: float = 0.1,
        num_workers: int = 4,
        seed: int = 42
        ):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        ),
    ])


    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
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
