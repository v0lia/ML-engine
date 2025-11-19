# get_data.py

# import sys
# from pathlib import Path

# import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.defaults import default_dataset_path
from src.utils.logger import get_logger

def get_dataset(mode_bool: bool, model_name: str):
    logger = get_logger()
    dataset = datasets.FashionMNIST(
        root=default_dataset_path.resolve(),
        train=mode_bool,
        download=True,
        transform=_get_transform(model_name))
        
    logger.info(f"Dataset: {dataset.__class__.__name__}")
    logger.info(f"Train size: {len(dataset)}" if mode_bool else f"Test size: {len(dataset)}")

    return dataset

def get_dataloader(mode: str, model_name: str, config: dict):
    logger = get_logger()
    batch_size = config.get("batch_size", 64)
    mode_bool = True if mode == "train" else False
    dataloader = DataLoader(get_dataset(mode_bool, model_name), batch_size=batch_size, shuffle=mode_bool)
    logger.info(f"Batchs size: {batch_size}. Total batches: {len(dataloader)}")
    return dataloader

def _get_transform(model_name):
    logger = get_logger()
    model_type = model_name.lower()
    if model_type in ["mlp", "lenet", "lenetimproved"]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # -> [-1, 1]
        ])
    elif model_type in ["resnet", "resnet18"]:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    else:
        error_text = f"[get_data._get_transforms] Unknown model_type: {model_type}"
        logger.error(error_text)
        raise ValueError(error_text)
    return transform
