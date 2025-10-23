# import torch
def train(dataloader, model, loss_fn, optimizer, config, device):
    model.model.to(device)
    model.train()   # set model to training mode
    epochs = config.get("epochs", 4)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} -------------")
        train_one_epoch(dataloader, model, loss_fn, optimizer, config, device)
        print("--------------------\n")
        # logger, metrics, visualize
    return

def train_one_epoch(dataloader, model, loss_fn, optimizer, config, device):
    # size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        y_prediction = model(X)   # forward pass
        # Backpropogation
        loss = loss_fn(y_prediction, y) # compute loss
        loss.backward() # compute grads wrt weights and biases: L(W), L(b)
        optimizer.step() # adjust weights and biases
        optimizer.zero_grad() # by default grads add up
        # logger, metrics
    return

# selective feedback - for each 100th batch
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch*batch_size+len(X)
        #     print(f"Loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")

# logger: на каждой итерации/эпохе.

# metrics: после прохода по батчу или эпохе, чтобы вычислить метрики.

# visualize: обычно после эпохи, когда собираются данные для графиков (или в конце всего тренинга).