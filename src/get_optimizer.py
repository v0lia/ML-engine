# import torch
import torch.optim as optim

def get_optimizer(model, config):
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(),lr=1e-3, betas=(0.9,0.999))
    try:
        optimizer_type = config["optimizer"]["type"]
    except KeyError:
        optimizer_type = "Adam"
    optimizer_class = getattr(optim, optimizer_type)
    optimizer = optimizer_class(model.parameters(), **config["params"])
    return optimizer