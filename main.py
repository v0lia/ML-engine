import argparse
import yaml
import torch
from pathlib import Path

from src import get_data
from src import get_model
from src import get_loss_function
from src import get_optimizer
from src import train
from src import evaluate

def get_config(config_path):
    path = Path(config_path).expanduser()
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Neural network train/eval pipeline")
    parser.add_argument("mode", choices=["train", "eval", "evaluate"],
                        help="Run mode: train or eval")
    parser.add_argument("model",
                        help="Name of a model from src/models/, e.g. CNN_v1")
    parser.add_argument("-c", "--config", "--cfg", default="src/config.yaml",
                        help="Path to YAML config file; default: src/config.yaml")
    return parser.parse_args()

def resolve_mode(mode: str):
    if mode == "train":
        return True
    elif mode in ["eval", "evaluate"]:
        return False
    else:   # will never happen unless argparser is changed
        raise ValueError(f"Unknown mode: {mode}. Available modes: train, eval.")

def main():
    args = parse_args()
    mode_bool = resolve_mode(args.mode)
    config = get_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    torch.manual_seed(config["seed"])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config["seed"])

    dataloader = get_data.get_dataloader(args.mode, config)
    model = get_model.get_model_class(args.model).to(device)
    loss_fn = get_loss_function.get_loss_function(config).to(device)
    optimizer = get_optimizer.get_optimizer(model, config)

    if mode_bool:   # train mode
        print(f"Starting training model {args.model}...")
        train.train(dataloader, model, loss_fn, optimizer, config, device)
        print(f"Finished training model {args.model}!")
    else:           # eval mode
        print(f"Starting evaluating model {args.model}...")
        evaluate.evaluate(dataloader, model, loss_fn, config, device)
        print(f"Finished evaluating model {args.model}!")

if __name__ == "__main__":
    main()

# TO DO
# 0) пройти урок по визуализации от PyTorch
# 1) структура вызова утилит в train.py
# 2) структура вызова утилит в evaluate.py
# 3) дополнить config.yaml параметрами частоты репорта 
# 4) утилита logger.py
# 5) утилита metrics.py
# 6) утилита visualize.py