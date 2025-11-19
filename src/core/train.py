# train.py

from datetime import datetime
# import torch

from src.utils import visualizer, metrics, timer
from src.utils.logger import get_logger
from src.utils.checkpoint import save_checkpoint
from src.utils.defaults import DATETIME_FORMAT

def train(dataloader, model, loss_fn, optimizer, config, device, writer, run_dir, start_epoch=0):
    logger = get_logger()
    logger.info(f"Starting training...\n")

    images, _ = next(iter(dataloader))  # images, labels
    visualizer.add_sample_grid(images, writer, tag="Train sample grid")
    visualizer.add_model_graph(model, images[:1], writer)
    visualizer.add_embedding(dataloader, writer, tag="Train data embedding")
        
    model.to(device)
    model.train()   # set model to training mode
    
    epochs = config.get("epochs", 4)

    global_batch_n = start_epoch * len(dataloader)  # global batch counter

    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"Start epoch {epoch+1}/{epochs}..................")
            with timer.Timer(name=f"{epoch+1}", logger=logger):
                epoch_loss, epoch_acc, global_batch_n = train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer, global_batch_n)
            
            visualizer.add_scalar(writer, "Train/loss/epoch", epoch_loss, epoch)
            visualizer.add_scalar(writer, "Train/accuracy/epoch", epoch_acc, epoch)
            #visualizer.add_prediction_grid(model, images.to(device), labels.to(device), writer, step=epoch)   # TBD someday
            
            model_name = model.__class__.__name__

            if epoch + 1 < epochs:  # save to run_dir/checkpoints
                save_checkpoint(run_dir=run_dir, model=model, optimizer=optimizer, epoch=epoch, config=config,
                            name = f"{model_name}_epoch_{epoch+1}_acc_{epoch_acc:.3f}_loss_{epoch_loss:.3f}")
            else:                   # save last checkpoint to default folder, e.g.: root/checkpoints
                save_checkpoint(run_dir=None, model=model, optimizer=optimizer, epoch=epoch, config=config,
                            name = f"{model_name}_acc_{epoch_acc:.3f}_{datetime.now():{DATETIME_FORMAT}}")
                        
            logger.info(f"End epoch {epoch+1} | Accuracy: {epoch_acc:.3f} | Average loss: {epoch_loss:.3f}")
            logger.info(f"----------------------------------------------------\n")
    except KeyboardInterrupt:
        logger.warning(f"Training interrupted by user (Ctlr+C). Saving unfinished checkpoint...")
        unfinished_path = save_checkpoint(run_dir=run_dir, model=model, optimizer=optimizer, epoch=epoch, config=config,
                            name=f"unfinished_{model_name}_epoch_{epoch+1}_acc_{epoch_acc:.3f}")
        logger.info(f"Saved unfinished checkpoint to: {unfinished_path.resolve()}")
        return

    logger.info(f"Finished training!")
    return

def train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer, global_batch_n):
    batch_losses = []
    batch_accuracies = []

    model.train()
    for (X,y) in dataloader:        # for batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True) # by default grads add up

        y_prediction = model(X)     # forward pass
                                    # Backpropogation
        loss = loss_fn(y_prediction, y) # compute loss
        loss.backward()                 # compute grads wrt weights and biases: L(W), L(b)
        optimizer.step()                # adjust weights and biases
        
        batch_loss = loss.item()
        batch_acc = metrics.compute_accuracy(y_prediction, y)

        batch_losses.append(batch_loss)
        batch_accuracies.append(batch_acc)

        visualizer.add_scalar(writer, "Train/loss/batch", batch_loss, global_batch_n)
        visualizer.add_scalar(writer, "Train/accuracy/batch", batch_acc, global_batch_n)
        global_batch_n += 1

        if global_batch_n % 100 == 1 and len(dataloader) > 0:
            print(f"...{(global_batch_n + 1) % len(dataloader)} / {len(dataloader)}...")

    epoch_loss = metrics.compute_avg_loss(batch_losses)
    epoch_acc =  metrics.compute_avg_accuracy(batch_accuracies)
    return epoch_loss, epoch_acc, global_batch_n 