# main.py

#!/usr/bin/env python3

import logging
import sys

import torch

from src.data.get_data import get_dataloader
from src.models.get_model import get_model
from src.utils.checkpoint import load_checkpoint
from src.loss.get_loss_function import get_loss_function
from src.optimizer.get_optimizer import get_optimizer
from src.core.train import train
from src.core.evaluate import evaluate
from src.utils import config as Config, argparser, seed, get_run_dir, logger as Logger, tensorboard_utils

def main():
    args = argparser.parse_args(Config.default_config_path)
    mode = argparser.resolve_mode(args.mode)

    path, is_checkpoint = argparser.resolve_checkpoint_or_model_path(mode_arg=args.mode, checkpoint_arg=args.checkpoint, model_arg=args.model, auto_latest=args.auto_latest)
    
    model_name = path.stem
    run_dir = get_run_dir.get_run_dir(mode, model_name)
    
    logger = Logger.setup_logger(name="main", level=logging.INFO, run_dir=run_dir)
    logger.info(f"{mode} {model_name}: initializing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: "{device}"')

    config, config_path = Config.get_config(args.config)
    seed.set_seeds(config, device)

    Config.copy_config_to_run_dir(config_path=config_path, run_dir=run_dir)
    argparser.copy_cmdline_to_run_dir(line=sys.argv, run_dir=run_dir)

    if is_checkpoint:
        logger.info(f"[MAIN] Loading checkpoint from: {path}")
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path=path, device=device)
    else:
        logger.info(f"[MAIN] Loading model from: {path}")
        model = get_model(model_name)
        optimizer = get_optimizer(model, config) 
        start_epoch = 0

    dataloader = get_dataloader(mode, model_name, config)
    
    loss_fn = get_loss_function(config, device)
    
    if mode == "train": # train mode  
        model.to(device)
        writer = tensorboard_utils.get_writer(run_dir)
        train(dataloader, model, loss_fn, optimizer, config, device, writer, run_dir, start_epoch=start_epoch)
    
    else:               # eval mode        
        model.to(device)        
        writer = tensorboard_utils.get_writer(run_dir)
        evaluate(dataloader, model, loss_fn, config, device, writer)        
    
    tensorboard_utils.close_writer(writer)

if __name__ == "__main__":
    main()
