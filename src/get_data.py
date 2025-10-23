# import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # -> [-1, 1]
    ])
    return transform

def get_dataset(mode_bool: bool):
    dataset = datasets.FashionMNIST(
        root=str(Path(__file__).resolve().parent.parent / "data"),
        train=mode_bool,
        download=True,
        transform=get_transform())
    return dataset

def get_dataloader(mode_bool: bool, config: dict):
    batch_size = config.get("batch_size", 64)
    dataloader = DataLoader(get_dataset(mode_bool), batch_size=batch_size, shuffle=mode_bool)
    return dataloader


