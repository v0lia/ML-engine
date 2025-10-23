import torch

def evaluate(dataloader, model, loss_fn, config, device):
    model.model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():   # no need to compute grads on test phase
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_prediction = model(X)
            
            test_loss += loss_fn(y_prediction,y).item()   # накапливаем ошибку от батча к батчу
            correct += (y_prediction.argmax(1) == y).type(torch.float).sum().item()   # сколько верных предсказаний в батче - накапливаем
    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f}")
    # logger, metrics, visualize
    return 

# logger: логируем финальные результаты (loss, accuracy, F1 и т.д.)

# metrics: вычисляем метрики по тестовому набору

# visualize: строим графики результатов (confusion matrix, примеры предсказаний)