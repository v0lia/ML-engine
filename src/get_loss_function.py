# import torch
import torch.nn as nn

def get_loss_function(config: dict):
    loss_function = config.get("loss_function", "CrossEntropyLoss")
    return getattr(nn, loss_function)()